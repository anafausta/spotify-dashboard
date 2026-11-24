@echo off
title Spotify Intelligence - Iniciar Servidor e Link Publico
echo ========================================================
echo       INICIANDO SPOTIFY INTELLIGENCE DASHBOARD
echo ========================================================
echo.
echo [1/2] Verificando Docker...
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo [AVISO] O Docker Desktop parece estar fechado.
    echo Por favor, abra o Docker Desktop para as predicoes funcionarem!
    echo.
) else (
    echo [OK] Docker esta ativo!
)

echo.
echo [2/2] Iniciando Servidor Node.js...
start "Spotify Servidor Local" cmd /k "cd /d C:\Users\Letic\.gemini\antigravity\scratch\spotify-dashboard && node server.js"

timeout /t 2 >nul

echo.
echo ========================================================
echo [OK] Servidor aberto em: http://localhost:3000
echo.
echo Iniciando Tunel Publico da Cloudflare...
echo Copie o link 'https://...trycloudflare.com' que aparecer abaixo!
echo ========================================================
echo.
C:\Users\Letic\.gemini\antigravity\scratch\cloudflared.exe tunnel --url http://localhost:3000
pause
