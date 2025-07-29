@echo off
echo ============================================================
echo iLLuMinator 4.9B Enhanced Training and Debugging System
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

REM Check if NVIDIA GPU is available
echo Checking NVIDIA GPU availability...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: NVIDIA GPU or drivers not detected
    echo Training will be very slow on CPU
    echo Please install NVIDIA drivers and CUDA toolkit
) else (
    echo ✅ NVIDIA GPU detected
    nvidia-smi
)

echo.
echo ============================================================
echo Setting up Python environment...
echo ============================================================

REM Install/upgrade pip
python -m pip install --upgrade pip

REM Install requirements
echo Installing enhanced requirements...
pip install -r requirements_enhanced.txt

REM Alternative CUDA installation if the above fails
if errorlevel 1 (
    echo.
    echo Trying alternative PyTorch installation...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers accelerate datasets wandb matplotlib seaborn psutil GPUtil tqdm numpy
)

echo.
echo ============================================================
echo Testing CUDA setup...
echo ============================================================

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"

echo.
echo ============================================================
echo Starting Training Process...
echo ============================================================

REM Set environment variables for optimal performance
set CUDA_VISIBLE_DEVICES=0
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6
set OMP_NUM_THREADS=4

REM Create necessary directories
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs

echo Starting enhanced 4.9B parameter training...
python train_4.9B_enhanced.py

echo.
echo ============================================================
echo Training completed! Starting debugging analysis...
echo ============================================================

python debug_4.9B_enhanced.py

echo.
echo ============================================================
echo Process completed!
echo ============================================================
echo.
echo Check the following locations for results:
echo - Checkpoints: checkpoints/
echo - Logs: training_4.9B_*.log
echo - Debug results: debug_output_*/
echo.

pause
