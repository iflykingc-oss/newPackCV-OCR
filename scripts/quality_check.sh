#!/usr/bin/env bash
# 代码质量检查脚本
# 用途: 提交前跑一遍，确保代码质量

set -e

echo "🔍 VibeCoding-OCR 代码质量检查"
echo "================================"
echo ""

# 1. 单元测试
echo "📋 [1/5] 单元测试..."
uv run pytest src/tests/unit/ -q --tb=short 2>&1 | tail -3

# 2. 集成测试
echo ""
echo "🔗 [2/5] 集成测试..."
uv run pytest src/tests/integration/ -q --tb=short 2>&1 | tail -3

# 3. Import规范检查
echo ""
echo "📦 [3/5] Import 规范检查 (禁止 src. 前缀)..."
bad_imports=$(grep -rn "from src\." src/ 2>/dev/null | grep -v __pycache__ | wc -l)
if [ "$bad_imports" -gt 0 ]; then
    echo "  ❌ 发现 $bad_imports 处违规 src. 前缀 import"
    grep -rn "from src\." src/ 2>/dev/null | grep -v __pycache__ | head -5
    exit 1
else
    echo "  ✅ 无违规"
fi

# 4. 节点函数签名检查
echo ""
echo "🔧 [4/5] 节点函数签名检查..."
bad_sigs=0
for f in src/graphs/nodes/*_node.py; do
    if ! grep -q "Runtime\[Context\]" "$f"; then
        echo "  ❌ $(basename $f): runtime 注解不规范"
        bad_sigs=$((bad_sigs + 1))
    fi
done
if [ "$bad_sigs" -gt 0 ]; then
    exit 1
else
    echo "  ✅ 全部节点函数签名规范"
fi

# 5. 配置文件完整性
echo ""
echo "📝 [5/5] 大模型配置完整性..."
bad_cfgs=0
for f in config/*.json; do
    python3 -c "
import json, sys
d = json.load(open('$f'))
sp = d.get('sp', '').strip()
up = d.get('up', '').strip()
if not sp or not up:
    sys.exit(1)
" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "  ❌ $f: sp 或 up 字段为空"
        bad_cfgs=$((bad_cfgs + 1))
    fi
done
if [ "$bad_cfgs" -gt 0 ]; then
    exit 1
else
    echo "  ✅ 全部配置完整"
fi

echo ""
echo "================================"
echo "✅ 全部检查通过！可以提交代码"
