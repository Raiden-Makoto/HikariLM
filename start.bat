@echo off
rem Start LightUsagi on Windows. Double-click this file, or run it with options: start.bat --port 9000
cd /d "%~dp0"
where py >nul 2>nul && (py -3 start.py %* & goto done)
where python >nul 2>nul && (python start.py %* & goto done)
echo Python is not installed. Get it from https://www.python.org/downloads/
:done
pause
