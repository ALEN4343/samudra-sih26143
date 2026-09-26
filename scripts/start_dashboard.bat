@echo off
rem ---------------------------------------------------------------------------
rem SAMUDRA - start the API and open the dashboard.
rem Double-click this file. It opens a server window (leave it open) and then
rem your default browser.
rem
rem Use 127.0.0.1, not localhost: uvicorn binds IPv4 only, and on Windows
rem "localhost" can resolve to the IPv6 ::1 first, which is refused.
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "URL=http://127.0.0.1:8000/app"

echo.
echo   SAMUDRA - Oil Discharge Attribution
echo   ------------------------------------
echo   repo   : %CD%
echo   python : %PY%
echo   url    : %URL%
echo.

rem Already running? Then just open the browser.
powershell -NoProfile -Command "try{ Invoke-WebRequest '%URL%' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 }catch{ exit 1 }"
if not errorlevel 1 (
  echo   Server already running - opening the browser.
  start "" "%URL%"
  ping -n 3 127.0.0.1 >nul
  exit /b 0
)

echo   Starting the API in a new window...
start "SAMUDRA API - leave this window open" "%PY%" -m samudra.api --port 8000

echo   Waiting for it to come up...
powershell -NoProfile -Command "for($i=0;$i -lt 60;$i++){ try{ Invoke-WebRequest '%URL%' -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0 }catch{ Start-Sleep -Milliseconds 500 } }; exit 1"

if errorlevel 1 (
  echo.
  echo   The server did not answer within 30 seconds.
  echo   Look at the "SAMUDRA API" window for the error, then try:
  echo       %PY% -m samudra.api --port 8000
  echo.
  pause
  exit /b 1
)

echo   Up. Opening %URL%
start "" "%URL%"
ping -n 3 127.0.0.1 >nul
endlocal

