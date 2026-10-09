@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0recompile.ps1" %*
