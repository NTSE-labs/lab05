@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 goto error
)
call ".venv\Scripts\activate.bat"
if not exist ".venv\.requirements_installed" (
    echo Installing project dependencies...
    python -m pip install --upgrade pip
    if errorlevel 1 goto error
    pip install -r requirements.txt
    if errorlevel 1 goto error
    echo installed>".venv\.requirements_installed"
)
python app.py
if errorlevel 1 goto error
exit /b 0
:error
echo.
echo An error occurred. See the messages above.
pause
exit /b 1
