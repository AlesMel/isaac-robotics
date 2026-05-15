"""Teacher-student distillation: frozen ToF MLP policy → camera NatureCNN student (racing).

Phase 1 (done separately):
    Train the ToF teacher to convergence:
        python scripts/skrl/train_racing.py --task=Isaac-Robots-Racing-Direct-v0 \\
            --num_envs 2048 --headless

Phase 2 (this script):
    Train the CNN student with combined PPO + distillation loss:
        total_loss = PPO_loss + λ * MSE(student_action_mean, teacher_action_mean)

    λ decays linearly from lambda_start → lambda_end over lambda_decay_frac of training,
    then pure PPO takes over.

Usage:
    python scripts/skrl/train_distill_racing.py \\
        --teacher-checkpoint logs/skrl/racing_direct/<run>/checkpoints/best_agent.pt \\
        --num_envs 512 --max_iterations 2000 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Distillation training: ToF teacher → CNN student (racing).")
parser.add_argument("--teacher-checkpoint", type=str, required=True,
                    help="Path to ToF teacher best_agent.pt checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-Robots-Racing-Direct-v0",
                    help="Teacher task ID (determines difficulty). Student uses the -Distill- variant.")
parser.add_argument("--num_envs", type=int, default=512,
                    help="Number of parallel student environments (camera-limited; <=512 recommended).")
parser.add_argument("--max_iterations", type=int, default=2000,
                    help="Number of PPO update iterations.")
parser.add_argument("--lambda-start", type=float, default=1.0,
                    help="Initial distillation loss weight.")
parser.add_argument("--lambda-end", type=float, default=0.0,
                    help="Final distillation loss weight (after decay).")
parser.add_argument("--lambda-decay-frac", type=float, default=0.7,
                    help="Fraction of iterations over which λ decays from start to end.")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True  # required for camera sensor
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.model_instantiators.torch import gaussian_model

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaac_robots.tasks  # noqa: F401

from isaac_robots.tasks.direct.isaac_labyrinth.agents.cnn_mlp_model import (
    NatureCnnPolicy,
    MlpCritic,
    PROPRIO_DIM,
    FRAME_STACK,
    CAM_H,
    CAM_W,
)


class _ProprioValueNet(MlpCritic):
    """Value network initialized with PROPRIO_DIM but accepts full policy obs at runtime.

    skrl PPO passes the full camera obs (16396D) to the value network via
    {"states": preprocessed_obs}.  This subclass slices the first PROPRIO_DIM
    dims so the underlying nn.Linear(PROPRIO_DIM, 256) always receives the
    correct input size.
    """

    def compute(self, inputs: dict, role: str):
        proprio = inputs["states"][:, :PROPRIO_DIM]
        return self.value_head(self.net(proprio)), {}
from isaac_robots.tasks.direct.isaac_racing.cfg import CRAZYFLIE_AI_CAMERA_64_CFG

# ── Constants ─────────────────────────────────────────────────────────────────
# Student uses the distill variant; teacher uses the standard task passed via --task.
DISTILL_TASK_SUFFIX = "-Distill"   # inserted before "-Direct-" to form the distill task ID
ROLLOUTS = 48       # must match skrl_ppo_cfg.yaml rollouts
TOF_DIM = 6         # 6 ToF rays (front/back/left/right/up/down)


def _make_distill_task_id(teacher_task_id: str) -> str:
    """Convert e.g. Isaac-Robots-Racing-Direct-v0 → Isaac-Robots-Racing-Distill-Direct-v0."""
    return teacher_task_id.replace("-Direct-", "-Distill-Direct-")


# ── Teacher loading ───────────────────────────────────────────────────────────

def build_teacher(teacher_task_id: str, checkpoint_path: str, device):
    """Instantiate the frozen ToF teacher policy and its observation preprocessor.

    Loads the model DIRECTLY from the checkpoint file — no second Isaac Lab simulation
    context is created (only one simulation is allowed per process).

    The teacher architecture is read from the task's skrl_ppo_cfg.yaml and reconstructed
    via skrl's gaussian_model instantiator, which guarantees exact weight-name alignment
    with the saved checkpoint.

    Returns:
        (teacher_policy, teacher_preprocessor) — both frozen / eval mode.
    """
    # ── Resolve YAML path from gym spec ──────────────────────────────────────
    spec = gym.spec(teacher_task_id)
    agent_cfg_entry = spec.kwargs["skrl_cfg_entry_point"]
    module_name, file_name = agent_cfg_entry.rsplit(":", 1)
    mod = importlib.import_module(module_name)
    yaml_path = os.path.join(os.path.dirname(mod.__file__), file_name)
    with open(yaml_path) as f:
        exp_cfg = yaml.safe_load(f)

    # ── Build teacher model (no gym.make / no Isaac Sim env) ─────────────────
    # Teacher obs = [proprio(12) | tof(6)] = 18D; actions = 4D
    teacher_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(PROPRIO_DIM + TOF_DIM,), dtype=np.float32
    )
    teacher_act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    policy_cfg = exp_cfg["models"]["policy"]
    net_cfg = policy_cfg["network"][0]
    n_layers = len(net_cfg["layers"])
    activations = net_cfg["activations"]
    if isinstance(activations, str):
        activations = [activations] * n_layers

    teacher_policy = gaussian_model(
        observation_space=teacher_obs_space,
        action_space=teacher_act_space,
        device=device,
        clip_actions=policy_cfg.get("clip_actions", False),
        clip_log_std=policy_cfg.get("clip_log_std", True),
        min_log_std=policy_cfg.get("min_log_std", -20.0),
        max_log_std=policy_cfg.get("max_log_std", 2.0),
        initial_log_std=policy_cfg.get("initial_log_std", 0.0),
        hiddens=net_cfg["layers"],
        hidden_activation=activations,
    )

    # ── Create obs preprocessor (RunningStandardScaler for 18D teacher input) ─
    teacher_preprocessor = RunningStandardScaler(size=PROPRIO_DIM + TOF_DIM, device=device)

    # ── Load weights + preprocessor state from checkpoint ────────────────────
    ckpt = torch.load(os.path.abspath(checkpoint_path), map_location=device, weights_only=False)
    print(f"[INFO] Checkpoint keys: {list(ckpt.keys())}")
    teacher_policy.load_state_dict(ckpt["policy"])
    teacher_policy.to(device)
    # skrl saves the actor obs preprocessor under "state_preprocessor" key
    if "state_preprocessor" in ckpt:
        teacher_preprocessor.load_state_dict(ckpt["state_preprocessor"])
        teacher_preprocessor.to(device)
    else:
        print("[WARN] No 'state_preprocessor' in checkpoint — teacher obs will not be normalized.")

    for p in teacher_policy.parameters():
        p.requires_grad_(False)
    teacher_policy.eval()
    teacher_preprocessor.eval()

    return teacher_policy, teacher_preprocessor


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    teacher_task_id = args_cli.task
    student_task_id = _make_distill_task_id(teacher_task_id)
    device_str = args_cli.device or "cuda"
    device = torch.device(device_str)

    print(f"[INFO] Teacher task: {teacher_task_id}")
    print(f"[INFO] Student task: {student_task_id}")

    # ── Student environment (camera + lidar both active) ──────────────────────
    env_cfg = parse_env_cfg(student_task_id, device=device_str, num_envs=args_cli.num_envs)
    env_cfg.camera = CRAZYFLIE_AI_CAMERA_64_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/body/ai_camera"
    )
    env_cfg.observation_space = PROPRIO_DIM + FRAME_STACK * CAM_H * CAM_W  # 16396
    env_cfg.seed = args_cli.seed

    env = gym.make(student_task_id, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    print(f"[INFO] obs_space={env.observation_space}, action_space={env.action_space}")

    # ── Frozen teacher (loaded directly — no second simulation context) ───────
    print(f"[INFO] Loading teacher from: {args_cli.teacher_checkpoint}")
    teacher_policy, teacher_preprocessor = build_teacher(teacher_task_id, args_cli.teacher_checkpoint, device)
    print("[INFO] Teacher loaded and frozen.")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_root = os.path.abspath(os.path.join("logs", "skrl", "racing_distill"))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_distill"
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    # SummaryWriter for distillation-specific metrics; PPO agent writes its own events
    distill_writer = SummaryWriter(os.path.join(log_dir, "distill"))
    print(f"[INFO] Logging to: {log_dir}")

    # ── Student models ────────────────────────────────────────────────────────
    student_policy = NatureCnnPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    ).to(device)

    # skrl PPO passes the full policy obs (16396D) to the value network.
    # _ProprioValueNet is initialized with PROPRIO_DIM and slices at runtime.
    proprio_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(PROPRIO_DIM,), dtype=np.float32
    )
    student_value = _ProprioValueNet(
        observation_space=proprio_space,
        action_space=env.action_space,
        device=device,
    ).to(device)

    # ── PPO agent (student) ───────────────────────────────────────────────────
    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=env.num_envs, device=device)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = ROLLOUTS
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 8
    cfg["discount_factor"] = 0.995
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveLR
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.01
    cfg["value_loss_scale"] = 2.0
    cfg["rewards_shaper"] = lambda r, *_: r * 0.1
    cfg["time_limit_bootstrap"] = True
    # Observation preprocessor for actor (stored as _state_preprocessor in skrl PPO)
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    # Value normalization
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # Logging / checkpointing (manual below)
    cfg["experiment"]["directory"] = log_dir
    cfg["experiment"]["experiment_name"] = ""
    cfg["experiment"]["write_interval"] = 200
    cfg["experiment"]["checkpoint_interval"] = 0

    student_agent = PPO(
        models={"policy": student_policy, "value": student_value},
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    student_agent.init()

    # Separate optimizer for distillation loss (does not interfere with PPO's optimizer)
    distill_optimizer = torch.optim.Adam(student_policy.parameters(), lr=1e-4)

    # ── Training loop ─────────────────────────────────────────────────────────
    TOTAL_STEPS = args_cli.max_iterations * ROLLOUTS
    lambda_decay_iters = max(1, int(args_cli.max_iterations * args_cli.lambda_decay_frac))

    obs, _ = env.reset()

    for iteration in range(args_cli.max_iterations):
        # Linear λ schedule: decays from lambda_start to lambda_end over decay window
        t = min(iteration, lambda_decay_iters) / lambda_decay_iters
        lam = args_cli.lambda_start + t * (args_cli.lambda_end - args_cli.lambda_start)
        lam = max(args_cli.lambda_end, lam)

        cam_obs_list: list[torch.Tensor] = []
        tof_obs_list: list[torch.Tensor] = []

        # ── Rollout collection ────────────────────────────────────────────────
        for step in range(ROLLOUTS):
            timestep = iteration * ROLLOUTS + step

            student_agent.pre_interaction(timestep=timestep, timesteps=TOTAL_STEPS)

            with torch.no_grad():
                outputs = student_agent.act(obs, timestep=timestep, timesteps=TOTAL_STEPS)
            actions = outputs[0]

            # obs is already the flat policy obs tensor (SkrlVecEnvWrapper extracts
            # observations["policy"] and returns it as a 2D tensor).
            cam_obs_list.append(obs.clone())

            next_obs, rewards, terminated, truncated, info = env.step(actions)
            tof_obs_list.append(info["tof_obs"].clone())  # (N, 6) from extras

            student_agent.record_transition(
                obs, actions, rewards, next_obs,
                terminated, truncated, info,
                timestep=timestep, timesteps=TOTAL_STEPS,
            )
            obs = next_obs

            # post_interaction triggers the PPO update at the last step of the rollout
            # (when (timestep + 1) % ROLLOUTS == 0) and handles TensorBoard logging.
            student_agent.post_interaction(timestep=timestep, timesteps=TOTAL_STEPS)

        # ── Distillation update ───────────────────────────────────────────────
        if lam > 1e-4:
            # (ROLLOUTS * N, obs_dim) and (ROLLOUTS * N, 6)
            cam_flat = torch.cat(cam_obs_list, dim=0)
            tof_flat = torch.cat(tof_obs_list, dim=0)

            # Reconstruct teacher input: [proprio(12) | tof(6)] = 18D
            # (proprio is the first PROPRIO_DIM dims of the camera observation)
            teacher_input = torch.cat([cam_flat[:, :PROPRIO_DIM], tof_flat], dim=-1)

            # Teacher forward (no grad — weights are frozen)
            with torch.no_grad():
                tof_pre = teacher_preprocessor(teacher_input.to(device), train=False).to(device)
                t_out = teacher_policy.act({"states": tof_pre}, role="policy")
                t_means = t_out[-1]["mean_actions"]  # (B, 4)

            # Student forward
            distill_optimizer.zero_grad()
            student_policy.train()
            cam_pre = student_agent._state_preprocessor(cam_flat, train=False)
            s_out = student_policy.act({"observations": cam_pre}, role="policy")
            s_means = s_out[-1]["mean_actions"]  # (B, 4)

            distill_loss = F.mse_loss(s_means, t_means)
            (lam * distill_loss).backward()
            torch.nn.utils.clip_grad_norm_(student_policy.parameters(), 1.0)
            distill_optimizer.step()

            global_step = (iteration + 1) * ROLLOUTS
            distill_writer.add_scalar("Distillation/lambda", lam, global_step)
            distill_writer.add_scalar("Distillation/mse_loss", distill_loss.item(), global_step)

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if (iteration + 1) % 50 == 0:
            ckpt_path = os.path.join(log_dir, "checkpoints", f"agent_{iteration + 1:05d}.pt")
            student_agent.save(ckpt_path)
            print(f"[INFO] Iter {iteration + 1}/{args_cli.max_iterations} — saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(log_dir, "checkpoints", "best_agent.pt")
    student_agent.save(final_path)
    print(f"[INFO] Training complete. Final checkpoint: {final_path}")

    distill_writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
