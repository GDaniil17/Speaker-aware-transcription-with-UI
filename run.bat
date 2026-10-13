@echo off
rem Windows: double-click this file, or run "run.bat --full" in a terminal. See README.md.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
set "code=%ERRORLEVEL%"
rem Keep the window open on failure so the error stays readable.
if not "%code%"=="0" pause
exit /b %code%
