@echo off
setlocal EnableExtensions DisableDelayedExpansion

set ROOT=%~dp0..\..\..
set CFG=%ROOT%\config\bot_config.json
set GRID=%ROOT%\src\tools\backtest_v2\grid.example.json
set OUT=%ROOT%\out_opt_v2

for /f "delims=" %%F in ('dir /b /o:-d "%ROOT%\scripts\*.xlsx" 2^>nul') do (
  set "DATA=%ROOT%\scripts\%%F"
  goto :found
)
for /f "delims=" %%F in ('dir /b /o:-d "%ROOT%\scripts\*.csv" 2^>nul') do (
  set "DATA=%ROOT%\scripts\%%F"
  goto :found
)

echo ERROR: No .xlsx or .csv file found in "%ROOT%\scripts"
pause
exit /b 1

:found
echo Using data file: "%DATA%"

cd /d "%ROOT%"

python -m tools.backtest_v2.optimize ^
  --data "%DATA%" ^
  --config "%CFG%" ^
  --grid "%GRID%" ^
  --out "%OUT%" ^
  --days 30

start "" "%OUT%"
echo.
echo Done. Results saved in: %OUT%
pause
endlocal
