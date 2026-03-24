@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON=python"
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%SCRIPT_DIR%build" mkdir "%SCRIPT_DIR%build"
"%PYTHON%" "%SCRIPT_DIR%build.py" --generator "MinGW Makefiles" --verbose --log-file "%SCRIPT_DIR%build\build.log" %*
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% NEQ 0 exit /b %EXIT_CODE%
echo Build completed successfully!
echo Full log written to %SCRIPT_DIR%build\build.log
