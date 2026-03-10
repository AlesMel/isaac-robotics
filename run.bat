@echo off
setlocal enabledelayedexpansion

:: Get the directory of the current script (includes trailing backslash)
set "ROOT_DIR=%~dp0"

:: If ISAACLAB_DIR is not set, check the first argument
if "%ISAACLAB_DIR%"=="" (
    if not "%~1"=="" if exist "%~1\" (
        set "ISAACLAB_DIR=%~1"
        shift
    ) else (
        echo Usage: %~nx0 C:\path\to\IsaacLab [train.py args...] >&2
        echo    or: set ISAACLAB_DIR=C:\path\to\IsaacLab ^& %~nx0 [train.py args...] >&2
        exit /b 1
    )
)
echo Using IsaacLab directory: %ISAACLAB_DIR%
:: Collect remaining arguments into a variable
set "ARGS="
:parse_args
if not "%~1"=="" (
    set "ARGS=!ARGS! %1"
    shift
    goto parse_args
)

:: Check for the Windows launcher script (.bat instead of .sh)
if exist "%ISAACLAB_DIR%\isaaclab.bat" (
    set "ISAACLAB_SCRIPT=%ISAACLAB_DIR%\isaaclab.bat"
) else if exist "%ISAACLAB_DIR%\isaac-sim.bat" (
    set "ISAACLAB_SCRIPT=%ISAACLAB_DIR%\isaac-sim.bat"
) else (
    echo Could not find an executable isaaclab launcher in %ISAACLAB_DIR% >&2
    echo Expected one of: isaaclab.bat or isaac-sim.bat >&2
    exit /b 1
)

:: Set PYTHONPATH (using Windows semicolon separator)
if "%PYTHONPATH%"=="" (
    set "PYTHONPATH=%ROOT_DIR%src"
) else (
    set "PYTHONPATH=%ROOT_DIR%src;%PYTHONPATH%"
)

:: Execute using 'call' (allows the batch script to yield to the Python script)
call "%ISAACLAB_SCRIPT%" -p "%ROOT_DIR%train.py" !ARGS!