@echo off
chcp 65001 >nul
echo ========================================
echo 增强向量化和语义搜索系统
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python环境
    echo 请确保已安装Python并添加到PATH
    pause
    exit /b 1
)

REM 设置工作目录
cd /d "%~dp0"

REM 显示菜单
:menu
echo 请选择操作:
echo 1. 安装依赖包
echo 2. 运行增强向量化 (Sentence-BERT)
echo 3. 运行增强向量化 (中文BERT)
echo 4. 运行增强向量化 (TF-IDF备用)
echo 5. 语义搜索演示
echo 6. 交互式搜索
echo 7. 退出
echo.
set /p choice="请输入选项 (1-7): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto vectorize_sbert
if "%choice%"=="3" goto vectorize_bert
if "%choice%"=="4" goto vectorize_tfidf
if "%choice%"=="5" goto demo
if "%choice%"=="6" goto interactive
if "%choice%"=="7" goto exit
echo 无效选项，请重新选择
goto menu

:install
echo.
echo 🔄 安装依赖包...
python enhanced_main.py --action install
echo.
echo ✅ 依赖包安装完成
pause
goto menu

:vectorize_sbert
echo.
echo 🔄 使用Sentence-BERT模型进行向量化...
python enhanced_main.py --action vectorize --model-type sentence_transformers
echo.
pause
goto menu

:vectorize_bert
echo.
echo 🔄 使用中文BERT模型进行向量化...
python enhanced_main.py --action vectorize --model-type transformers
echo.
pause
goto menu

:vectorize_tfidf
echo.
echo 🔄 使用TF-IDF进行向量化...
python enhanced_main.py --action vectorize --model-type tfidf
echo.
pause
goto menu

:demo
echo.
echo 🔍 运行语义搜索演示...
python enhanced_main.py --action demo
echo.
pause
goto menu

:interactive
echo.
echo 🔍 启动交互式搜索...
python enhanced_main.py --action interactive
echo.
pause
goto menu

:exit
echo.
echo 👋 感谢使用!
pause
exit /b 0
