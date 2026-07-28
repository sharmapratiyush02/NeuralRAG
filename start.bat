@echo off
setlocal EnableExtensions

title NeuralRAG Quick Start
cd /d "%~dp0"

echo.
echo ================================================================
echo  NeuralRAG - Windows Quick Start
echo ================================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PY=python"
    ) else (
        echo [ERROR] Python 3 was not found.
        echo Install Python 3.10+ from https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
)

%PY% --version
if errorlevel 1 (
    echo [ERROR] Python is installed but could not be started.
    pause
    exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [1/5] Creating virtual environment...
    %PY% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Virtual environment already exists.
)

call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

echo.
echo [2/5] Installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo [3/5] Preparing project folders...
if not exist "data" mkdir "data"
if not exist "data\chroma_db" mkdir "data\chroma_db"
if not exist "data\documents" mkdir "data\documents"
if not exist "data\evaluation" mkdir "data\evaluation"
if not exist "data\evaluation\results" mkdir "data\evaluation\results"
if not exist "models" mkdir "models"
if not exist "models\fine_tuned_embedder" mkdir "models\fine_tuned_embedder"

if not exist "data\documents\sample_rag_overview.txt" (
    echo Creating sample document...
    > "data\documents\sample_rag_overview.txt" echo # Introduction to Retrieval-Augmented Generation
    >> "data\documents\sample_rag_overview.txt" echo.
    >> "data\documents\sample_rag_overview.txt" echo Retrieval-Augmented Generation, or RAG, improves language model answers by retrieving relevant documents before generating a response.
    >> "data\documents\sample_rag_overview.txt" echo.
    >> "data\documents\sample_rag_overview.txt" echo Core components include document ingestion, chunking, neural embeddings, hybrid retrieval, reranking, grounded generation, and evaluation.
)

if "%ANTHROPIC_API_KEY%"=="" if "%OPENAI_API_KEY%"=="" (
    echo.
    echo [WARN] No ANTHROPIC_API_KEY or OPENAI_API_KEY was found.
    echo The project may use local/demo behavior unless you add an API key.
)

echo.
echo [4/5] Ingesting documents...
python scripts\ingest.py --input data\documents\ --chunking recursive
if errorlevel 1 (
    echo [WARN] Ingestion failed or will complete when the API starts.
)

echo.
echo [5/5] Starting NeuralRAG...
echo API:      http://localhost:8000
echo Docs:     http://localhost:8000/docs
echo Frontend: http://localhost:3000
echo.

start "NeuralRAG Frontend" cmd /k "cd /d ""%~dp0frontend"" && call ""%~dp0venv\Scripts\activate.bat"" && python -m http.server 3000"
timeout /t 2 /nobreak >nul
start "" "http://localhost:3000"

echo Starting API server in this window...
echo Press Ctrl+C to stop it.
echo.
cd /d "%~dp0backend"
python api.py

echo.
echo NeuralRAG API stopped.
pause
