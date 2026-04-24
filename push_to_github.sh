#!/bin/bash

# PackCV-OCR 推送到 GitHub 脚本
# 使用方法: ./push_to_github.sh <YOUR_GITHUB_TOKEN>

echo "=========================================="
echo "  PackCV-OCR GitHub 推送工具"
echo "=========================================="
echo ""

# 检查是否提供了 token
if [ -z "$1" ]; then
    echo "❌ 错误: 请提供 GitHub Personal Access Token"
    echo ""
    echo "使用方法:"
    echo "  ./push_to_github.sh <YOUR_GITHUB_TOKEN>"
    echo ""
    echo "获取 Token 步骤:"
    echo "  1. 访问: https://github.com/settings/tokens"
    echo "  2. 点击 'Generate new token' → 'Generate new token (classic)'"
    echo "  3. 勾选 'repo' 权限"
    echo "  4. 点击 'Generate token'"
    echo "  5. 复制生成的 token"
    echo ""
    exit 1
fi

TOKEN=$1

echo "📦 项目信息:"
echo "  仓库地址: https://github.com/iflykingc-oss/newPackCV-OCR.git"
echo "  当前分支: main"
echo ""

# 检查 Git 状态
echo "📊 检查 Git 状态..."
git status --short

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Git 状态检查失败"
    exit 1
fi

echo ""
echo "✅ Git 状态检查完成"
echo ""

# 配置远程仓库 URL（包含 token）
echo "🔧 配置远程仓库..."
git remote set-url origin https://${TOKEN}@github.com/iflykingc-oss/newPackCV-OCR.git

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 配置远程仓库失败"
    exit 1
fi

echo "✅ 远程仓库配置完成"
echo ""

# 推送代码
echo "🚀 开始推送到 GitHub..."
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✅ 推送成功！"
    echo "=========================================="
    echo ""
    echo "📦 项目地址:"
    echo "  https://github.com/iflykingc-oss/newPackCV-OCR"
    echo ""
    echo "🔗 快速访问:"
    echo "  查看代码: https://github.com/iflykingc-oss/newPackCV-OCR/tree/main"
    echo "  提交历史: https://github.com/iflykingc-oss/newPackCV-OCR/commits/main"
    echo "  Issues:   https://github.com/iflykingc-oss/newPackCV-OCR/issues"
    echo ""
    echo "🎉 PackCV-OCR 融合算法系统已成功部署到 GitHub！"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "  ❌ 推送失败"
    echo "=========================================="
    echo ""
    echo "可能的原因:"
    echo "  1. Token 无效或已过期"
    echo "  2. 网络连接问题"
    echo "  3. 仓库权限不足"
    echo ""
    echo "解决方法:"
    echo "  1. 检查 Token 是否正确"
    echo "  2. 访问 https://github.com/settings/tokens 重新生成 Token"
    echo "  3. 确认你有该仓库的推送权限"
    echo ""
    exit 1
fi
