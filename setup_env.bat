@echo off
echo ======================================================
echo   Gemini 335 机械臂视觉环境一键配置工具
echo ======================================================

:: 1. 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.11 并勾选 Add to PATH。
    pause
    exit
)

:: 2. 创建虚拟环境
echo [1/4] 正在创建虚拟环境 (venv)...
python -m venv venv

:: 3. 激活环境并安装标准库
echo [2/4] 正在安装标准依赖库 (requirements.txt)...
call .\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. 安装离线奥比中光 SDK
echo [3/4] 正在安装奥比中光离线驱动 (WHL)...
:: 这里会自动寻找文件夹下的 whl 文件进行安装
pip install pyorbbecsdk2*.whl

:: 5. 完成
echo [4/4] 环境配置完成！
echo ------------------------------------------------------
echo 现在您可以运行运行 python Camera_Catch.py 了。
echo ------------------------------------------------------
pause