@echo off
echo ========================================
echo     运行 Alpaca 数据集构建器
echo ========================================
echo.

echo 正在构建 Alpaca 格式训练数据集...
uv run alpaca_dataset_builder.py

echo.
echo 数据集构建完成！
echo 输出文件位置: output/alpaca_dataset/
echo.
pause
