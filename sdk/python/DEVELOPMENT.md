# PackCV-OCR Python SDK

轻量级 Python 客户端,零强制依赖 (核心功能仅用标准库 `urllib`)。

## 运行测试

```bash
cd sdk/python
pip install -e ".[dev]"
pytest tests/ -v
```

## 构建

```bash
python -m build
```

## 发布到 PyPI

```bash
python -m twine upload dist/*
```
