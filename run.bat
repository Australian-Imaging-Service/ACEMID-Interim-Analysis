@echo off
REM One-command runner (Windows).
REM   run.bat <tiles_dir> [out_dir] [extra run_inference.py flags...]
REM Auto-creates a local .venv and installs requirements.txt on first run.
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo usage: run.bat ^<tiles_dir^> [out_dir] [extra flags...]
  exit /b 1
)
set "TILES=%~1"
shift
set "OUT=results"
set "FIRST=%~1"
if not "%FIRST%"=="" (
  echo %FIRST% | findstr /b "--" >nul
  if errorlevel 1 ( set "OUT=%FIRST%" & shift )
)

REM Collect any remaining flags.
set "EXTRA="
:collect
if not "%~1"=="" ( set "EXTRA=%EXTRA% %~1" & shift & goto collect )

REM Use current Python if deps present, else build a local venv once.
python -c "import torch, open_clip, PIL, pandas, numpy" >nul 2>&1
if errorlevel 1 (
  if not exist .venv (
    echo [env] creating local .venv and installing requirements ^(first run only^)...
    python -m venv .venv
    .venv\Scripts\pip install --upgrade pip >nul
    .venv\Scripts\pip install -r requirements.txt
  )
  set "PY=.venv\Scripts\python"
) else (
  set "PY=python"
)

echo [run] scoring tiles in: %TILES%  -^>  %OUT%\
%PY% run_inference.py --tiles-dir "%TILES%" --out-dir "%OUT%" %EXTRA%
echo [done] see %OUT%\per_tile_predictions.csv and %OUT%\per_patient_severity.csv
endlocal
