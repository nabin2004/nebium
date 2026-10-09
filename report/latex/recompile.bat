@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0recompile.ps1" %*
