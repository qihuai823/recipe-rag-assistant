@echo off
rem Start the web demo and open the browser.
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" goto found
set "PY=C:\Users\lenovo\Desktop\all-in-rag-main\all-in-rag-main\.venv\Scripts\python.exe"
if exist "%PY%" goto found
set "PY=python"

:found
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo Starting web demo, browser will open in a few seconds...
start "" cmd /c "timeout /t 10 >nul & start http://127.0.0.1:8000"
"%PY%" server.py
