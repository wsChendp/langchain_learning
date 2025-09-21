"""
GLM-4 兼容版本：使用普通提示词生成结构化输出
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel

# 使用 GLM-4 模型
model = ChatOpenAI(
    model='glm-4-0520',
    temperature=0.6,
    api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
    base_url='https://open.bigmodel.cn/api/paas/v4',
)

class Classification(BaseModel):
    sentiment: str  # 情感分类，字符串类型
    aggressive: int = Field(
        description="攻击性，整数类型，范围0-10",
    )
    language: str = Field(
        description="语言，字符串类型",
    )

def classify_text_with_glm4(text: str) -> Classification:
    """使用 GLM-4 进行文本分类"""
    
    # 创建提示词模板
    prompt = ChatPromptTemplate.from_template(
        """请分析以下文本，提取以下信息：
1. sentiment: 情感分类（如：积极、消极、中性）
2. aggressive: 攻击性程度（0-10的整数，0表示无攻击性，10表示极度攻击性）
3. language: 语言类型（如：中文、英文等）

请严格按照以下JSON格式返回，不要包含任何其他内容：
{{
    "sentiment": "情感分类",
    "aggressive": 攻击性数字,
    "language": "语言类型"
}}

文本: {input}"""
    )
    
    # 构建链
    chain = prompt | model
    
    # 调用模型
    response = chain.invoke({'input': text})
    content = response.content
    
    print(f"🔍 模型原始响应: {content}")
    
    # 提取JSON
    try:
        # 查找JSON对象
        if "{" in content and "}" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_str = content[json_start:json_end]
        else:
            json_str = content.strip()
        
        print(f"📄 提取的JSON: {json_str}")
        
        # 解析JSON
        data = json.loads(json_str)
        
        # 创建Classification对象
        classification = Classification(**data)
        return classification
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        # 返回默认值
        return Classification(
            sentiment="未知",
            aggressive=0,
            language="未知"
        )
    except Exception as e:
        print(f"❌ 分类失败: {e}")
        return Classification(
            sentiment="未知",
            aggressive=0,
            language="未知"
        )

# 测试
text = '我今天心情很好，但是有点生气，因为有人骂我傻逼。我说的都是中文，你能听懂吗？'

print("🚀 开始文本分类...")
result = classify_text_with_glm4(text)

print(f"\n✅ 分类结果:")
print(f"  情感: {result.sentiment}")
print(f"  攻击性: {result.aggressive}/10")
print(f"  语言: {result.language}")
