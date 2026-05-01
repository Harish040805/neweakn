@echo off
title EAKN Launcher

echo Starting EAKN Backend...
:: Change directory to your project folder
cd /d "D:\eaknnewproject"

:: Start the Python backend in a new minimized window
start /min python update.py

echo Waiting for server to initialize...
timeout /t 3 /nobreak >nul

echo Launching EAKN Frontend...
:: Opens your specific HTML file in the default browser
start "" "D:\eaknnewproject\eakn.html"

exit