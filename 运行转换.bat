@echo off
chcp 65001
echo ========================================
echo 芯片知识库 Markdown 到 JSON 转换工具
echo ========================================
echo.

REM 设置路径
set INPUT_DIR=markdown
set OUTPUT_DIR=output

echo 输入目录: %INPUT_DIR%
echo 输出目录: %OUTPUT_DIR%
echo.

REM 检查输入目录是否存在
if not exist "%INPUT_DIR%" (
    echo 错误: 输入目录不存在: %INPUT_DIR%
    echo 请确保 markdown 目录存在
    pause
    exit /b 1
)

echo 开始转换处理...
echo.

REM 运行Python脚本
python main.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 转换完成! 
    echo ========================================
    echo.
    echo 输出文件位置:
    echo - JSON数据: %OUTPUT_DIR%\json\
    echo - 分析报告: %OUTPUT_DIR%\analysis\
    echo - 向量化数据: %OUTPUT_DIR%\vectorization\
    echo.
    echo 你现在可以:
    echo 1. 查看分析报告了解数据分布
    echo 2. 使用向量化数据进行嵌入处理
    echo 3. 基于JSON数据进行标签和分片处理
    echo.
) else (
    echo.
    echo ========================================
    echo 转换失败!
    echo ========================================
    echo 请检查错误信息并重试
    echo.
)

pause
