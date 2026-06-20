"""场景Schema基类"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FieldDef(BaseModel):
    """字段定义"""
    name: str = Field(..., description="字段名")
    description: str = Field(..., description="字段说明")
    required: bool = Field(default=False, description="是否必填")
    field_type: str = Field(default="str", description="字段类型: str/list/dict/number/date")
    example: Optional[str] = Field(default=None, description="示例值")
    validation_rules: List[str] = Field(default=[], description="校验规则")


class BaseSchema(BaseModel):
    """场景Schema基类 - 所有场景继承此类"""
    scenario_type: str = Field(..., description="场景类型标识")
    scenario_name: str = Field(..., description="场景中文名")
    description: str = Field(..., description="场景描述")
    fields: List[FieldDef] = Field(..., description="字段定义列表")
    system_prompt: str = Field(..., description="场景LLM系统提示词")
    user_prompt_template: str = Field(..., description="用户提示词模板（Jinja2）")

    def get_required_fields(self) -> List[str]:
        return [f.name for f in self.fields if f.required]

    def get_optional_fields(self) -> List[str]:
        return [f.name for f in self.fields if not f.required]

    def build_output_template(self) -> str:
        """构建JSON输出模板"""
        lines = ["{"]
        for i, f in enumerate(self.fields):
            comma = "," if i < len(self.fields) - 1 else ""
            default = "[]" if f.field_type == "list" else "{}" if f.field_type == "dict" else "null"
            lines.append(f'  "{f.name}": {default}{comma}  # {f.description}')
        lines.append("}")
        return "\n".join(lines)