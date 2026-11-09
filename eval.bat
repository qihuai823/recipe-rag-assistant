@echo off
rem Run evaluation. Examples:
rem   eval.bat
rem   eval.bat --no-filter
rem   eval.bat --limit 5
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
"%PY%" eval\run_eval.py %*
