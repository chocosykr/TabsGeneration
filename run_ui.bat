@echo off
cd /d "%~dp0"
echo [TabsGeneration] Launching Interactive Audio ^& Tab Inspector...

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" prototype\ui_server.py %*
) else (
    python prototype\ui_server.py %*
)

pause
