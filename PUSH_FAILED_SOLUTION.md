# 🔐 GitHub 推送失败 - 解决方案

## 问题说明

推送失败，错误信息：`Authentication failed for 'https://github.com/iflykingc-oss/newPackCV-OCR.git/'`

**可能的原因**：
1. GitHub 的 Secret Scanning 阻止了 token 直接在 URL 中使用
2. Token 的权限配置可能有问题
3. Git 认证方式配置不正确

---

## ✅ 解决方案

### 方案一：在本地手动推送（最可靠）

由于当前环境的限制，建议你在**本地电脑**上执行以下步骤：

#### 步骤 1: 克隆或下载项目

```bash
# 选项 A: 如果你有 SSH 访问权限
git clone git@github.com:iflykingc-oss/newPackCV-OCR.git

# 选项 B: 如果需要 token
git clone https://<YOUR_TOKEN>@github.com/iflykingc-oss/newPackCV-OCR.git
```

#### 步骤 2: 复制项目文件

将当前项目目录的所有文件复制到你克隆的仓库中。

#### 步骤 3: 提交并推送

```bash
cd newPackCV-OCR

# 添加所有文件
git add .

# 提交
git commit -m "feat: PackCV-OCR 融合算法系统完整实现"

# 推送
git push -u origin main
```

---

### 方案二：使用 GitHub CLI（推荐）

如果你安装了 `gh` CLI 工具，可以使用它来推送：

```bash
# 1. 登录 GitHub
gh auth login

# 2. 克隆仓库
gh repo clone iflykingc-oss/newPackCV-OCR

# 3. 复制文件并推送
cd newPackCV-OCR
git add .
git commit -m "feat: PackCV-OCR 融合算法系统完整实现"
git push -u origin main
```

---

### 方案三：使用 SSH 密钥

#### 步骤 1: 生成 SSH 密钥

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

#### 步骤 2: 添加 SSH 密钥到 GitHub

```bash
# 复制公钥
cat ~/.ssh/id_ed25519.pub

# 然后访问 https://github.com/settings/keys 添加
```

#### 步骤 3: 推送代码

```bash
git remote set-url origin git@github.com:iflykingc-oss/newPackCV-OCR.git
git push -u origin main
```

---

## 📦 项目文件列表

你需要推送以下文件：

### 核心代码
```
src/
├── graphs/
│   ├── state.py
│   ├── graph.py
│   ├── packcv_graph.py
│   └── nodes/
│       ├── cv_detection_node.py
│       ├── roi_segmentation_node.py
│       ├── parallel_processing_node.py
│       ├── alert_engine_node.py
│       └── report_generation_node.py
└── storage/
    ├── schema.sql
    ├── models.py
    └── database.py
```

### 配置文件
```
config/
├── model_extract_llm_cfg.json
├── correct_text_llm_cfg.json
└── qa_answer_llm_cfg.json
```

### 文档
```
README.md
AGENTS.md
PUSH_GUIDE.md
PUSH_TO_GITHUB.md
READY_TO_PUSH.md
```

### 其他
```
pyproject.toml
.gitignore
push_to_github.sh
```

---

## 🚀 快速上传指南

如果你在本地电脑上操作，可以快速执行以下命令：

```bash
# 1. 创建临时目录
mkdir /tmp/packcv-ocr
cd /tmp/packcv-ocr

# 2. 初始化 Git
git init

# 3. 添加远程仓库（使用你的 token）
git remote add origin https://<YOUR_TOKEN>@github.com/iflykingc-oss/newPackCV-OCR.git

# 4. 复制所有项目文件
# （将当前项目的所有文件复制到 /tmp/packcv-ocr/）

# 5. 添加文件
git add .

# 6. 提交
git commit -m "feat: PackCV-OCR 融合算法系统完整实现"

# 7. 推送
git push -u origin main
```

---

## 📞 需要帮助？

如果遇到问题：

1. **检查 Token 权限**: 确保你的 token 有 `repo` 权限
2. **验证仓库**: 访问 https://github.com/iflykingc-oss/newPackCV-OCR 确认仓库存在
3. **查看日志**: 运行 `GIT_TRACE=1 GIT_CURL_VERBOSE=1 git push -u origin main` 查看详细日志

---

## 🎯 替代方案

如果你无法推送，可以考虑：

1. **创建 Pull Request**: Fork 仓库，然后创建 PR
2. **使用 GitHub Web 界面**: 手动上传文件到仓库
3. **联系仓库管理员**: 请求管理员为你添加推送权限

---

**建议**: 在你的本地电脑上使用**方案一**或**方案二**，这样更可靠且不会遇到认证问题。
