@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: پیدا کردن مسیر دقیق پوشه اصلی پروژه (NDS-Flow-Scalper-Bot)
:: فرض بر این است که این فایل در src\tools\backtest_v2 قرار دارد
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%..\..\.."
set "ROOT=%cd%"

:: تنظیم مسیرهای حیاتی
set "CFG=%ROOT%\config\bot_config.json"
set "GRID=%ROOT%\src\tools\backtest_v2\grid.example.json"
set "OUT=%ROOT%\out_opt_v2"
set "DATA_DIR=%ROOT%\scripts"

echo Searching for data files in: "%DATA_DIR%"

:: جستجو برای فایل اکسل
if exist "%DATA_DIR%\*.xlsx" (
    for /f "delims=" %%F in ('dir /b /o:-d "%DATA_DIR%\*.xlsx"') do (
        set "DATA=%DATA_DIR%\%%F"
        goto :found
    )
)

:: جستجو برای فایل CSV (اگر اکسل پیدا نشد)
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
echo --------------------------------------------------

:: رفتن به پوشه اصلی پروژه برای اجرای صحیح ماژول پایتون
cd /d "%ROOT%"

:: اجرای بهینه‌ساز با ماژولار پایتون
python -m src.tools.backtest_v2.optimize ^
  --data "%DATA%" ^
  --config "%CFG%" ^
  --grid "%GRID%" ^
  --out "%OUT%" ^
  --days 10

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Backtest optimization failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

:: باز کردن پوشه نتایج
if exist "%OUT%" start "" "%OUT%"

echo.
echo Done. Results saved in: %OUT%
pause
endlocal
