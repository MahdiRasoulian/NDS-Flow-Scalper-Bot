@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: -----------------------------
:: Locate project root
:: -----------------------------
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%..\..\.."
set "ROOT=%cd%"

:: -----------------------------
:: Critical paths
:: -----------------------------
set "CFG=%ROOT%\config\bot_config.json"
set "GRID=%ROOT%\src\tools\backtest_v2\grid.example.json"
set "OUT=%ROOT%\out_opt_v2"
set "DATA_DIR=%ROOT%\scripts"

:: -----------------------------
:: Tunables (EDIT HERE)
:: -----------------------------
set "DAYS=25"
set "ROWS="
set "WARMUP=450"
set "LOG_LEVEL=INFO"
set "EXPORT_DEBUG=1"

:: Spread/slippage (optional)
set "SPREAD="
set "SLIPPAGE="

echo Searching for data files in: "%DATA_DIR%"

:: Find newest .xlsx
if exist "%DATA_DIR%\*.xlsx" (
    for /f "delims=" %%F in ('dir /b /o:-d "%DATA_DIR%\*.xlsx"') do (
        set "DATA=%DATA_DIR%\%%F"
        goto :found
    )
)

:: Find newest .csv
if exist "%DATA_DIR%\*.csv" (
    for /f "delims=" %%F in ('dir /b /o:-d "%DATA_DIR%\*.csv"') do (
        set "DATA=%DATA_DIR%\%%F"
        goto :found
    )
)

echo ERROR: No .xlsx or .csv file found in "%DATA_DIR%"
pause
exit /b 1

:found
echo --------------------------------------------------
echo [OK] Using data file: "%DATA%"
echo [OK] Config file:    "%CFG%"
echo [OK] Grid file:      "%GRID%"
echo [OK] Output dir:     "%OUT%"
echo --------------------------------------------------

cd /d "%ROOT%"

:: -----------------------------
:: Build args
:: -----------------------------
set "ARGS=--data "%DATA%" --config "%CFG%" --grid "%GRID%" --out "%OUT%" --days %DAYS% --warmup %WARMUP% --log-level %LOG_LEVEL%"

if not "%ROWS%"=="" (
  set "ARGS=%ARGS% --rows %ROWS%"
)

if not "%SPREAD%"=="" (
  set "ARGS=%ARGS% --spread %SPREAD%"
)

if not "%SLIPPAGE%"=="" (
  set "ARGS=%ARGS% --slippage %SLIPPAGE%"
)

if "%EXPORT_DEBUG%"=="1" (
  set "ARGS=%ARGS% --export-debug"
)

echo [RUN] python -m src.tools.backtest_v2.optimize %ARGS%
echo --------------------------------------------------

python -m src.tools.backtest_v2.optimize %ARGS%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Backtest optimization failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

if exist "%OUT%" start "" "%OUT%"

echo.
echo Done. Results saved in: %OUT%
pause
endlocal
