@echo off
cls
REM === Lancement du serveur
REM
Start "Environment" /b venv\Scripts\activate
REM
REM === Lancement de Open WebUI
REM
START "Open WebUI" /b open-webui serve --port 2026
REM