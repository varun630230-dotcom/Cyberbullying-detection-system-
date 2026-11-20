@echo off
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"
set "PYTHON_EXE=C:\Users\varun\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Could not find the configured Python interpreter:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)
start "Cyberbullying Detection Studio" "%PYTHON_EXE%" -m uvicorn app:app --host 127.0.0.1 --port 5000
ping 127.0.0.1 -n 3 > nul
start "" http://127.0.0.1:5000/
