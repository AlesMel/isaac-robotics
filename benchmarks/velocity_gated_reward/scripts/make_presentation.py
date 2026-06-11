"""Generate a simple blue-and-white PowerPoint presentation for the velocity-gated reward benchmark."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Colour palette ──────────────────────────────────────────────────────────
DARK_BLUE  = RGBColor(0x0D, 0x2B, 0x55)   # slide background / title bg
MID_BLUE   = RGBColor(0x1A, 0x5F, 0xA8)   # accent / header bar
LIGHT_BLUE = RGBColor(0xD6, 0xE8, 0xF8)   # subtle tint rows / boxes
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT     = RGBColor(0x2E, 0xA8, 0xE4)   # bright highlight
SLIDE_W    = Inches(13.33)
SLIDE_H    = Inches(7.5)


def set_font(run, size, bold=False, color=WHITE):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_background(slide, color=DARK_BLUE):
    from pptx.util import Emu
    bg = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        0, 0, SLIDE_W, SLIDE_H,
    )
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    sp = bg._element
    sp.getparent().remove(sp)
    slide.shapes._spTree.insert(2, sp)


def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, left, top, width, height, text, size, bold=False,
                color=WHITE, align=PP_ALIGN.LEFT, wrap=True):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    txBox.word_wrap = wrap
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    set_font(run, size, bold, color)
    return txBox


def add_header_bar(slide, title):
    """Dark blue top bar + white title."""
    add_rect(slide, 0, 0, SLIDE_W, Inches(1.15), DARK_BLUE)
    add_textbox(slide, Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.8),
                title, 28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)


def add_accent_bar(slide):
    """Thin MID_BLUE bottom bar."""
    add_rect(slide, 0, Inches(7.2), SLIDE_W, Inches(0.3), MID_BLUE)


# ────────────────────────────────────────────────────────────────────────────
#  Slide factories
# ────────────────────────────────────────────────────────────────────────────

def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_background(slide, DARK_BLUE)

    # Accent bar at top
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.08), ACCENT)

    # Central box
    add_rect(slide, Inches(1.0), Inches(1.5), Inches(11.33), Inches(4.0), MID_BLUE)

    add_textbox(slide, Inches(1.2), Inches(1.8), Inches(11.0), Inches(1.2),
                "Velocity-Gated Reward Shaping",
                40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    add_textbox(slide, Inches(1.2), Inches(3.0), Inches(11.0), Inches(0.9),
                "for PPO Reach-and-Hold Manipulation",
                30, bold=False, color=LIGHT_BLUE, align=PP_ALIGN.CENTER)

    add_textbox(slide, Inches(1.2), Inches(4.0), Inches(11.0), Inches(0.7),
                "Benchmark Study  ·  Isaac Lab  ·  Ales Melichar  ·  2026",
                18, bold=False, color=RGBColor(0xAA, 0xCC, 0xEE), align=PP_ALIGN.CENTER)

    add_rect(slide, 0, Inches(7.2), SLIDE_W, Inches(0.3), ACCENT)
    return slide


def slide_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, DARK_BLUE)
    add_header_bar(slide, "The Problem: Reach + Hold in Continuous Control")
    add_accent_bar(slide)

    body_y = Inches(1.3)
    add_textbox(slide, Inches(0.5), body_y, Inches(12.3), Inches(0.5),
                "Training a manipulator to reach a goal is easy. Holding there is not.",
                20, color=LIGHT_BLUE)

    # Two failure mode boxes
    for i, (label, body, col) in enumerate([
        ("Vanilla tanh  →  Limit Cycle",
         "Non-zero gradient at d = 0 keeps pushing the policy.\n"
         "Result: visible oscillation that never settles.",
         RGBColor(0x1A, 0x3A, 0x6A)),
        ("Gaussian  →  Late-stage Drift",
         "Zero gradient at d = 0 gives no centering pull.\n"
         "Result: PPO noise disperses the policy after convergence.",
         RGBColor(0x0D, 0x45, 0x7A)),
    ]):
        x = Inches(0.5) + i * Inches(6.4)
        add_rect(slide, x, Inches(2.1), Inches(6.0), Inches(3.2), col)
        add_textbox(slide, x + Inches(0.15), Inches(2.2), Inches(5.7), Inches(0.6),
                    label, 18, bold=True, color=ACCENT)
        add_textbox(slide, x + Inches(0.15), Inches(2.9), Inches(5.7), Inches(2.2),
                    body, 16, color=WHITE)

    add_textbox(slide, Inches(0.5), Inches(5.5), Inches(12.3), Inches(0.5),
                "Both failure modes stem from the shape of the reward kernel near the goal.",
                17, color=LIGHT_BLUE, align=PP_ALIGN.CENTER)
    return slide


def slide_solution(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, DARK_BLUE)
    add_header_bar(slide, "Our Solution: Velocity-Gated Tanh Kernel")
    add_accent_bar(slide)

    add_textbox(slide, Inches(0.5), Inches(1.3), Inches(12.3), Inches(0.45),
                "Keep the centering pull of tanh, but gate it by how much the arm is moving.",
                19, color=LIGHT_BLUE)

    # Formula box
    add_rect(slide, Inches(0.5), Inches(1.9), Inches(12.3), Inches(1.9), MID_BLUE)
    add_textbox(slide, Inches(0.7), Inches(2.0), Inches(12.0), Inches(0.5),
                "R(d, q̇)  =  lifted  ×  (1 − tanh(d / σ))  ×  G(d, q̇)", 21, bold=True, color=WHITE)
    add_textbox(slide, Inches(0.7), Inches(2.55), Inches(12.0), Inches(1.0),
                "G(d, q̇)  =  clip(1 − ‖q̇‖_arm / v_thresh , 0, 1)   if  d < r_neighborhood\n"
                "            =  1                                                   otherwise",
                17, color=LIGHT_BLUE)

    # Three property bullets
    props = [
        ("Outside goal zone", "Identical to vanilla tanh — reach behaviour unchanged."),
        ("Inside goal zone",  "Gate penalises the arm's own corrective oscillation directly."),
        ("Result",            "Centering pull preserved (no drift) + oscillation discouraged (no limit cycle)."),
    ]
    for i, (label, desc) in enumerate(props):
        y = Inches(4.05) + i * Inches(0.82)
        add_rect(slide, Inches(0.5), y, Inches(0.06), Inches(0.5), ACCENT)
        add_textbox(slide, Inches(0.7), y - Inches(0.02), Inches(3.0), Inches(0.55),
                    label, 16, bold=True, color=ACCENT)
        add_textbox(slide, Inches(3.8), y - Inches(0.02), Inches(9.0), Inches(0.55),
                    desc, 16, color=WHITE)

    return slide


def slide_kernels(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, DARK_BLUE)
    add_header_bar(slide, "Three Kernels Under Comparison")
    add_accent_bar(slide)

    headers = ["Kernel", "Formula", "Gradient at d=0", "Failure Mode"]
    rows = [
        ["tanh\n(baseline)", "1 − tanh(d / σ)", "−1/σ  ≠ 0", "Limit cycle"],
        ["Gaussian", "exp(−(d/σ)²)", "0", "Late-stage drift"],
        ["Velocity-gated tanh\n(ours)", "tanh × G(d, q̇)", "−1/σ × G  (preserved)", "None observed"],
    ]
    col_ws = [Inches(2.3), Inches(3.7), Inches(3.0), Inches(3.0)]
    col_xs = [Inches(0.4), Inches(2.75), Inches(6.5), Inches(9.55)]
    row_ys = [Inches(1.35), Inches(2.3), Inches(3.6), Inches(4.7)]
    row_hs = [Inches(0.7), Inches(1.1), Inches(1.0), Inches(1.0)]

    # Header row
    for c, (hdr, x, w) in enumerate(zip(headers, col_xs, col_ws)):
        add_rect(slide, x, row_ys[0], w - Inches(0.05), row_hs[0], MID_BLUE)
        add_textbox(slide, x + Inches(0.05), row_ys[0] + Inches(0.07), w - Inches(0.1), row_hs[0],
                    hdr, 15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    row_colors = [
        RGBColor(0x12, 0x35, 0x60),
        RGBColor(0x0D, 0x3D, 0x6E),
        RGBColor(0x08, 0x4A, 0x80),
    ]
    fail_colors = [
        RGBColor(0xCC, 0x44, 0x44),
        RGBColor(0xCC, 0x88, 0x22),
        RGBColor(0x22, 0xAA, 0x55),
    ]

    for r, (row, bg) in enumerate(zip(rows, row_colors)):
        ry = row_ys[r + 1]
        rh = row_hs[r + 1]
        for c, (cell, x, w) in enumerate(zip(row, col_xs, col_ws)):
            add_rect(slide, x, ry, w - Inches(0.05), rh, bg)
            txt_color = fail_colors[r] if c == 3 else WHITE
            bold = (c == 3)
            add_textbox(slide, x + Inches(0.05), ry + Inches(0.05), w - Inches(0.1), rh,
                        cell, 14, bold=bold, color=txt_color, align=PP_ALIGN.CENTER)

    return slide


def slide_hypotheses(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, DARK_BLUE)
    add_header_bar(slide, "Testable Hypotheses")
    add_accent_bar(slide)

    hyps = [
        ("H1a  —  Performance",
         "peak_reward(vel-gated) ≥ peak_reward(tanh) − ε\n"
         "Not significantly worse than the baseline (Welch's t, α = 0.05)."),
        ("H1b  —  No Limit Cycle",
         "hold_joint_vel_l2(vel-gated) < hold_joint_vel_l2(tanh)\n"
         "with Cohen's d > 0.8 (large effect size)."),
        ("H1c  —  No Drift",
         "drift_ratio(vel-gated) ≤ 5 %  AND  < drift_ratio(Gaussian)\n"
         "where drift_ratio = (peak − final) / peak over last 10 % of training."),
        ("H1d  —  Generalisation",
         "H1a–c hold for both Franka Panda (7-DoF) and UR3e + Hand-E (6-DoF)."),
    ]
    for i, (title, body) in enumerate(hyps):
        y = Inches(1.4) + i * Inches(1.4)
        add_rect(slide, Inches(0.4), y, Inches(0.07), Inches(0.9), ACCENT)
        add_textbox(slide, Inches(0.6), y, Inches(12.0), Inches(0.45),
                    title, 17, bold=True, color=ACCENT)
        add_textbox(slide, Inches(0.6), y + Inches(0.45), Inches(12.0), Inches(0.75),
                    body, 15, color=WHITE)
    return slide


def slide_methodology(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, DARK_BLUE)
    add_header_bar(slide, "Experimental Design")
    add_accent_bar(slide)

    # Left column
    add_rect(slide, Inches(0.4), Inches(1.3), Inches(5.9), Inches(5.5), MID_BLUE)
    add_textbox(slide, Inches(0.55), Inches(1.4), Inches(5.6), Inches(0.45),
                "Setup", 18, bold=True, color=ACCENT)
    setup = (
        "• 3 kernels  ×  5 seeds  ×  2 tasks  =  30 runs\n"
        "• Only the reward kernel differs — all PPO\n"
        "  hyper-parameters kept identical across runs\n\n"
        "Tasks\n"
        "  ① Franka Panda lift (7-DoF, 36 k timesteps)\n"
        "  ② UR3e + Hand-E lift (6-DoF, 300 k timesteps)\n\n"
        "Hardware\n"
        "  RTX 4070 Ti · 12 GB VRAM\n"
        "  ≈ 10 GPU-hours for full default sweep"
    )
    add_textbox(slide, Inches(0.55), Inches(1.9), Inches(5.6), Inches(4.6),
                setup, 15, color=WHITE)

    # Right column
    add_rect(slide, Inches(6.7), Inches(1.3), Inches(6.2), Inches(5.5), MID_BLUE)
    add_textbox(slide, Inches(6.85), Inches(1.4), Inches(6.0), Inches(0.45),
                "Metrics", 18, bold=True, color=ACCENT)
    metrics = (
        "Training scalars (TensorBoard)\n"
        "  • peak_reward — max episode reward\n"
        "  • final_reward — mean over last 10 % steps\n"
        "  • drift_ratio — (peak − final) / peak\n\n"
        "Deterministic eval rollouts (100 eps / seed)\n"
        "  • hold_joint_vel_l2 — RMS arm velocity\n"
        "    during hold window (last 2 s)\n"
        "  • hold_ee_z_std — EE vertical oscillation\n"
        "  • success_rate — cube within 3 cm of goal\n\n"
        "Statistics\n"
        "  Welch's t-test (α = 0.05) + Cohen's d"
    )
    add_textbox(slide, Inches(6.85), Inches(1.9), Inches(6.0), Inches(4.6),
                metrics, 15, color=WHITE)
    return slide


def slide_status(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, DARK_BLUE)
    add_header_bar(slide, "Current Status & Next Steps")
    add_accent_bar(slide)

    items_done = [
        "Benchmark infrastructure complete (sweep orchestrator, metrics, analysis)",
        "All three kernels implemented and unit-tested",
        "Franka lift training running (seed 0 / tanh — first smoke-test run active)",
        "Sweep config checked in (sweep_default.yaml, sweep_quick.yaml)",
        "UR3e + Hand-E environment registered and PPO config verified",
    ]
    items_next = [
        "Complete full sweep: 30 runs × Franka + UR3e platforms",
        "Run deterministic eval rollouts on all checkpoints",
        "Generate publication-ready plots and LaTeX tables",
        "Write results.md summary and technical report section",
        "Optional: stretch task Franka Reach + hyper-parameter ablation",
    ]

    add_rect(slide, Inches(0.4), Inches(1.3), Inches(5.9), Inches(5.5),
             RGBColor(0x0A, 0x38, 0x60))
    add_textbox(slide, Inches(0.55), Inches(1.35), Inches(5.7), Inches(0.45),
                "Completed", 17, bold=True, color=RGBColor(0x22, 0xCC, 0x66))
    for i, item in enumerate(items_done):
        add_textbox(slide, Inches(0.65), Inches(1.85) + i * Inches(0.84),
                    Inches(5.6), Inches(0.7),
                    "✓  " + item, 13, color=WHITE)

    add_rect(slide, Inches(6.7), Inches(1.3), Inches(6.2), Inches(5.5),
             RGBColor(0x08, 0x2B, 0x50))
    add_textbox(slide, Inches(6.85), Inches(1.35), Inches(6.0), Inches(0.45),
                "Next Steps", 17, bold=True, color=ACCENT)
    for i, item in enumerate(items_next):
        add_textbox(slide, Inches(6.95), Inches(1.85) + i * Inches(0.84),
                    Inches(5.9), Inches(0.7),
                    "→  " + item, 13, color=WHITE)

    return slide


def slide_summary(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, DARK_BLUE)
    add_header_bar(slide, "Summary")
    add_accent_bar(slide)

    points = [
        ("Problem",
         "Standard tanh reward causes limit cycles; Gaussian causes drift — "
         "both make it hard to hold at a goal."),
        ("Idea",
         "Multiply tanh by a velocity gate inside the goal neighbourhood: "
         "preserves the centering pull, penalises the arm's own oscillation."),
        ("Hypothesis",
         "Velocity-gated tanh achieves better hold stability and no late-stage drift "
         "without sacrificing peak reward, across Franka and UR3e."),
        ("Approach",
         "Controlled sweep: 3 kernels × 5 seeds × 2 tasks, identical PPO hyper-params, "
         "statistical comparison with Welch's t + Cohen's d."),
    ]
    for i, (label, body) in enumerate(points):
        y = Inches(1.4) + i * Inches(1.35)
        add_rect(slide, Inches(0.4), y, Inches(2.5), Inches(1.0), MID_BLUE)
        add_textbox(slide, Inches(0.5), y + Inches(0.2), Inches(2.3), Inches(0.6),
                    label, 18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_textbox(slide, Inches(3.15), y + Inches(0.1), Inches(9.8), Inches(0.85),
                    body, 16, color=WHITE)
        # connector line
        add_rect(slide, Inches(2.95), y + Inches(0.43), Inches(0.2), Inches(0.04), ACCENT)

    add_textbox(slide, Inches(0.4), Inches(6.8), Inches(12.5), Inches(0.4),
                "github.com/<TBD>/velocity_gated_reward  ·  Ales Melichar  ·  2026",
                13, color=RGBColor(0x88, 0xAA, 0xCC), align=PP_ALIGN.CENTER)
    return slide


# ────────────────────────────────────────────────────────────────────────────
#  Build
# ────────────────────────────────────────────────────────────────────────────

def build():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    slide_title(prs)
    slide_problem(prs)
    slide_solution(prs)
    slide_kernels(prs)
    slide_hypotheses(prs)
    slide_methodology(prs)
    slide_status(prs)
    slide_summary(prs)

    out = "/home/urkui-3/Documents/isaac-robotics/benchmarks/velocity_gated_reward/velgate_benchmark_presentation.pptx"
    prs.save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    build()
