import csv
from typing import Type

import requests

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import langgraph.prebuilt.chat_agent_executor as chat_agent_executor


def find_code(csv_file, district_name) -> str:
    # 读取CSV文件并创建地区到代码的映射
    district_map = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            district_code = row['district_id'].strip()
            district = row['district'].strip()
            if district not in district_map:
                district_map[district] = district_code

    return district_map.get(district_name, "未知地区")

class WeatherInputArgs(BaseModel):
    # 这个schema是传递给模型的输入
    location: str = Field(..., description='用于查询天气的位置信息')

class WeatherTool(BaseTool):
    name: str = 'weather_tool'
    description: str = '可以查询任意位置的当前天气的情况'
    args_schema: Type[WeatherInputArgs] = WeatherInputArgs

    def _run(self, location: str) -> str:
        district_code = find_code('weather_district_id.csv', location)
        weather_info = f"{location}（代码: {district_code}）的天气是晴天，温度25摄氏度。"
        url = f"https://api.map.baidu.com/weather/v1/?district_id={district_code}&data_type=now&ak=9AAIhEfDDboaJFkOiwn4krENvR9e9Fel"

        resp = requests.get(url=url).json()
        text = resp['result']['now']['text']
        temp = resp['result']['now']['temp']
        rh = resp['result']['now']['rh']
        feels_like = resp['result']['now']['feels_like']
        wind_dir = resp['result']['now']['wind_dir']
        wind_class = resp['result']['now']['wind_class']

        return f"{location}（代码: {district_code}）的天气是{text}，温度{temp}摄氏度，体感温度{feels_like}摄氏度，相对湿度{rh}%，风向{wind_dir}，风力{wind_class}级。"

if __name__ == "__main__":

    model = ChatOpenAI(
        model='glm-4-0520',
        temperature=0.6,
        api_key='06ca1c42545b44b2a3bb85531c7024a8.bDEKFcPHfhhYvm1Q',
        base_url='https://open.bigmodel.cn/api/paas/v4',
    )

    tools = [WeatherTool()]
    agent_executor = chat_agent_executor.create_tool_calling_executor(model,tools)

    resp = agent_executor.invoke({'messages': [HumanMessage(content='中国的首都是哪个城市')]})
    print(resp['messages'])

    reps2 = agent_executor.invoke({'messages': [HumanMessage(content='深圳的天气怎么样')]})
    print(reps2['messages'])

    print(reps2['messages'][2].content)
