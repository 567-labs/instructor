# Instructor：大语言模型（LLM）的结构化输出

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

从任意大语言模型（LLM）获取可靠的 JSON 输出。基于 Pydantic 构建，提供数据验证、类型安全与 IDE 智能提示支持。

```python
import instructor
from pydantic import BaseModel


# 定义您期望的数据结构
class User(BaseModel):
    name: str
    age: int


# 从自然语言中直接提取
client = instructor.from_provider("openai/gpt-4o-mini")
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)

print(user)  # User(name='John', age=25)
```

**就是这么简单。** 无需手动解析 JSON，无需编写繁琐的错误处理，也无需手动重试。只需定义一个数据模型，即可直接获得结构化数据。

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/567-labs/instructor?style=flat-square)](https://github.com/567-labs/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

> **需要快速提取结构化数据时请使用 Instructor；需要构建智能体（Agents）时请选择 PydanticAI。** Instructor 让模式优先（Schema-first）的工作流保持简单且低成本。如果您的应用需要更丰富的智能体运行时、内置可观测性或可共享的追踪记录（Traces），欢迎尝试 [PydanticAI](https://ai.pydantic.dev/)。PydanticAI 是 Pydantic 团队打造的官方智能体运行时，在复用相同 Pydantic 模型的同时，提供了强类型工具、可重放数据集、评估（Evals）以及生产级仪表盘。欢迎阅读 [PydanticAI 文档](https://ai.pydantic.dev/)，了解它如何扩展 Instructor 风格的工作流。

## 为什么选择 Instructor？

从大语言模型中获取结构化数据往往十分繁琐。您通常需要：

1. 编写复杂的 JSON Schema
2. 处理数据验证错误
3. 对失败的提取请求进行重试
4. 解析非结构化的响应内容
5. 适配不同模型提供商的 API 差异

**Instructor 通过一个简洁统一的接口为您搞定上述所有问题：**

<table>
<tr>
<td><b>不使用 Instructor</b></td>
<td><b>使用 Instructor</b></td>
</tr>
<tr>
<td>

```python
response = openai.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "..."}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "extract_user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
            },
        }
    ],
)

# 解析响应
tool_call = response.choices[0].message.tool_calls[0]
user_data = json.loads(tool_call.function.arguments)

# 手动验证
if "name" not in user_data:
    # 处理错误...
    pass
```

</td>
<td>

```python
client = instructor.from_provider("openai/gpt-5.4-mini")

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)

# 大功告成！user 已通过验证并具备完整类型
```

</td>
</tr>
</table>

## 秒级安装

```bash
pip install instructor
```

或者使用您的包管理器安装：
```bash
uv add instructor
poetry add instructor
```

## 支持所有主流模型提供商

使用完全相同的代码对接任意 LLM 提供商：

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama（本地模型）
client = instructor.from_provider("ollama/llama3.2")

# 直接传入 API 密钥（无需配置环境变量）
client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
client = instructor.from_provider("anthropic/claude-3-5-sonnet", api_key="sk-ant-...")
client = instructor.from_provider("groq/llama-3.1-8b-instant", api_key="gsk_...")

# 所有提供商均使用完全一致的 API！
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## 生产就绪的核心特性

### 自动重试（Automatic retries）

当数据验证失败时，会自动附带错误信息进行重试：

```python
from pydantic import BaseModel, field_validator


class User(BaseModel):
    name: str
    age: int

    @field_validator('age')
    def validate_age(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v


# 当验证失败时，Instructor 会自动进行重试
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
    max_retries=3,
)
```

### 流式传输支持（Streaming support）

在对象生成过程中实时流式获取部分对象（Partial objects）：

```python
from instructor import Partial

for partial_user in client.chat.completions.create(
    response_model=Partial[User],
    messages=[{"role": "user", "content": "..."}],
    stream=True,
):
    print(partial_user)
    # User(name=None, age=None)
    # User(name="John", age=None)
    # User(name="John", age=25)
```

### 嵌套对象（Nested objects）

轻松提取复杂的嵌套数据结构：

```python
from typing import List


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]


# Instructor 会自动处理嵌套对象
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## 广泛的生产环境应用

深受超过 100,000 名构建 AI 应用的开发者与企业信赖：

- **每月 300 万+ 下载量**
- **10,000+ GitHub Stars**
- **1,000+ 社区贡献者**

正在使用 Instructor 的企业包括来自 OpenAI、Google、Microsoft、AWS 的团队以及众多 YC 创业公司。

## 快速开始

### 基础提取

从任意文本中提取结构化数据：

```python
from pydantic import BaseModel
import instructor

client = instructor.from_provider("openai/gpt-4o-mini")


class Product(BaseModel):
    name: str
    price: float
    in_stock: bool


product = client.chat.completions.create(
    response_model=Product,
    messages=[{"role": "user", "content": "iPhone 15 Pro, $999, available now"}],
)

print(product)
# Product(name='iPhone 15 Pro', price=999.0, in_stock=True)
```

### 多语言支持

Instructor 简洁易用的 API 已支持多种编程语言：

- [Python](https://python.useinstructor.com) - 原生官方版本
- [TypeScript](https://js.useinstructor.com) - 完整的 TypeScript 支持
- [Ruby](https://ruby.useinstructor.com) - Ruby 实现版本
- [Go](https://go.useinstructor.com) - Go 实现版本
- [Elixir](https://hex.pm/packages/instructor) - Elixir 实现版本
- [Rust](https://rust.useinstructor.com) - Rust 实现版本

### 了解更多

- [官方文档](https://python.useinstructor.com) - 全面详尽的使用指南
- [示例代码](https://python.useinstructor.com/examples/) - 开箱即用的代码范例
- [技术博客](https://python.useinstructor.com/blog/) - 教程与最佳实践
- [Discord 社区](https://discord.gg/bD9YE9JArw) - 获取社区支持与帮助

## 为什么选择 Instructor 而非其他方案？

**对比原生 JSON 模式（Raw JSON mode）**：Instructor 提供自动验证、失败重试、流式传输以及嵌套对象支持，无需手写 JSON Schema。

**对比 LangChain/LlamaIndex**：Instructor 专注于做好一件事——结构化数据提取。它更轻量、更快速、也更容易调试。

**对比自研方案**：经过数万名开发者的生产环境实战检验，妥善处理了各种您可能尚未遇到的边界情况。

## 参与贡献

我们非常欢迎社区贡献！欢迎浏览我们的 [Good First Issues](https://github.com/567-labs/instructor/labels/good%20first%20issue) 开始参与。

## 开源许可证

MIT 许可证 - 详见 [LICENSE](https://github.com/567-labs/instructor/blob/main/LICENSE)。

---

<p align="center">
由 Instructor 社区共同构建。特别感谢 <a href="https://twitter.com/jxnlco">Jason Liu</a> 以及所有<a href="https://github.com/567-labs/instructor/graphs/contributors">贡献者</a>。
</p>

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月26日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
