@echo off
cd /d "D:\eaknnewproject"

:: Start python without a new window
start /b python update.py

:: Short wait for the server
timeout /t 5 /nobreak >nul

:: Open the frontend
start "" "D:\eaknnewproject\eakn.html"
exit