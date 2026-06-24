# 贡献指南

欢迎贡献 VibeCoding-OCR！请遵循以下流程：

## 1. 提交 Issue

提交前请先搜索 [现有 Issue](https://github.com/vibecoding-ocr/vibecoding-ocr/issues)。

**Bug Report** 请用 [Bug 报告模板](.github/ISSUE_TEMPLATE/bug_report.md)，包含：
- 复现步骤
- 期望行为
- 实际行为
- 截图/日志

**Feature Request** 请用 [功能请求模板](.github/ISSUE_TEMPLATE/feature_request.md)。

## 2. 提交代码

### 流程
1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交代码：`git commit -m "feat: 添加某某功能"`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

### 提交规范（Conventional Commits）

```
feat: 新功能
fix: Bug 修复
docs: 文档变更
refactor: 重构
test: 测试
chore: 构建/工具变更
```

### 代码规范

- Python 3.12+
- 类型注解：必须
- 文档字符串：Google 风格，必须
- 测试覆盖：核心逻辑必须测
- Lint：ruff + pyright
- 单文件 ≤ 500 行（拆分建议）

### 测试

```bash
# 跑全部测试
uv run pytest src/tests/ -v

# 跑单个文件
uv run pytest src/tests/unit/test_scenario_registry.py -v

# 覆盖率
uv run pytest src/tests/ --cov=src --cov-report=term-missing
```

## 3. 文档贡献

- `docs/` 目录下添加新文档
- 使用 Markdown，标题分层清晰
- 代码示例必须可运行
- 提交 PR 时链接到相关 Issue

## 4. 行为准则

参考 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

## 5. 许可证

贡献的代码将采用 [Apache 2.0 许可证](LICENSE)。
