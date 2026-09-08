#!/bin/bash
# 双击这个文件 = 打开 torch API 速查工具（终端窗口里输入 API 名）
cd "$(dirname "$0")"
/Users/liuzejiang/miniconda3/envs/dl/bin/python torch_api_guide.py
echo ""
echo "（按回车关闭窗口）"
read -r _
