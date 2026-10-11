@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run run.bat once to create the virtual environment and install dependencies.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pytest -v
pause
