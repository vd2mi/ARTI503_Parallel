@echo off
setlocal
cd /d "%~dp0"
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY (
  echo Python 3 is not installed on this computer.
  echo Install it from https://www.python.org/downloads/
  echo During setup tick the box "Add python.exe to PATH", then double-click this file again.
  echo.
  pause
  exit /b 1
)
%PY% run.py %*
echo.
pause
