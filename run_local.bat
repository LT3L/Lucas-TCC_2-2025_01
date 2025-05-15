@echo off

REM Diretórios esperados
set DATASET_DIR=%CD%\app\datasets
set OUTPUT_DIR=%CD%\app\datasets_and_models_output

REM Criar diretórios se não existirem
if not exist "%DATASET_DIR%" mkdir "%DATASET_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Ativar ambiente virtual se existir
if exist ".venv\Scripts\activate.bat" (
    echo ==^> Ativando ambiente virtual
    call .venv\Scripts\activate.bat
)

REM Instalar dependências se necessário
echo ==^> Verificando dependências
pip install -r requirements.txt

REM Adicionar o diretório atual ao PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%CD%

REM Executar o script principal
echo ==^> Executando o script principal
python biblioteca/memory_models/linear_memory_train.py 
python biblioteca/memory_models/linear_memory_test.py