# 开发者指南

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/vibecoding-ocr/vibecoding-ocr.git
cd vibecoding-ocr

# 2. 安装依赖 (uv)
uv sync

# 3. 运行测试
uv run pytest src/tests/ -v

# 4. 启动服务
uv run python src/web_server.py
```

## 添加自定义节点

每个节点都是 `src/graphs/nodes/{name}_node.py` 下的独立文件。

### 1. 定义 State (在 `src/graphs/state.py`)

```python
class MyNodeInput(BaseModel):
    package_image: File = Field(..., description="输入图片")
    
class MyNodeOutput(BaseModel):
    result: str = Field(..., description="提取结果")
```

### 2. 实现节点函数

```python
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import MyNodeInput, MyNodeOutput


def my_node(
    state: MyNodeInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> MyNodeOutput:
    """
    title: 我的节点
    desc: 节点功能描述
    integrations: 使用的技能名
    """
    ctx = runtime.context
    # 业务逻辑
    return MyNodeOutput(result="done")
```

### 3. 注册到图 (在 `src/graphs/graph.py`)

```python
from graphs.nodes.my_node import my_node
builder.add_node("my_node", my_node)
builder.add_edge("upstream_node", "my_node")
```

## 添加新场景

1. 在 `src/utils/scenario_schemas/` 创建 `{name}.py`
2. 继承 `BaseSchema` 定义字段
3. 在 `registry.py` 注册
4. 创建 `config/{name}_extract_llm_cfg.json` LLM 配置
5. 在 `config_manager.py` 的 `_SCENARIO_LLM_MAP` 添加映射

详细见 [场景开发指南](./development/custom-scenario.md)。

## 添加自定义引擎

1. 在 `src/utils/ocr_engines/` 创建 `{name}_engine.py`
2. 继承 `BaseOCREngine` 实现 `recognize()` 方法
3. 在 `smart_router.py` 的 `_ENGINES` 列表注册
4. 优先级数字越小越优先 (0 = 最高)
