@echo off
title Spotify Intelligence - Dashboard & Classificador de Hits
echo ========================================================
echo       INICIANDO SPOTIFY INTELLIGENCE DASHBOARD
echo ========================================================
echo.
cd /d "%~dp0"

echo [1/2] Iniciando Servidor Python FastAPI...
start "Spotify Servidor" cmd /k "cd /d "%~dp0" && python app.py"

timeout /t 3 >nul

echo [2/2] Abrindo navegador...
start http://localhost:7860

echo.
echo ========================================================
echo [OK] Dashboard em execucao em: http://localhost:7860
echo ========================================================
echo.
pause
