@echo off
rem One-click launcher: works from Explorer double-click or from any terminal.
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" goto found
set "PY=C:\Users\lenovo\Desktop\all-in-rag-main\all-in-rag-main\.venv\Scripts\python.exe"
if exist "%PY%" goto found
set "PY=python"
echo [warn] project venv not found, falling back to "python" on PATH

:found
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
"%PY%" %*
