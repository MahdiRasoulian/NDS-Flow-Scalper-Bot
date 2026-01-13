@echo off
setlocal EnableExtensions DisableDelayedExpansion

set ROOT=E:\BourseAnalysis\nds-scalping-v6
set CFG=%ROOT%\config\bot_config.json
set GRID=%ROOT%\config\grid.json
set OUT=%ROOT%\out_opt

REM پیدا کردن آخرین فایل xlsx/csv داخل scripts
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

python -m src.tools.backtest.optimize ^
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
