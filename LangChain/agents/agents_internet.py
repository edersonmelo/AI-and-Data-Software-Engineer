import chainlit as cl
from langchain.agents import AgentExecutor, load_tools, create_openai_functions_agent
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.tools import YouTubeSearchTool, Tool
import requests
from pytube import YouTube
from dotenv import load_dotenv
import requests
import os
import json
import base64
import os

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv(key="OPENAI_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.getenv(key="GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv(key="GOOGLE_API_KEY")

messages = []

llm = ChatOpenAI(model_name="gpt-4-1106-preview")

tools = load_tools(
    ["arxiv", "google-search"],
)


tools.append(YouTubeSearchTool())

prompt = hub.pull("hwchase17/openai-functions-agent")

llm = ChatOpenAI(model='gpt-4-1106-preview')

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@cl.on_message 
async def main(message: cl.Message):

    msg = cl.Message(content="")
    await msg.send()
    
    response = await agent_executor.ainvoke({"input": message.content})
    msg.content = response["output"]
    
    await msg.update()
    
