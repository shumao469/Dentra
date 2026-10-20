@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Launch_Slicer_Review.ps1"
if errorlevel 1 pause
