from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import YouTubeSearchTool
from typing import Any
from langchain.pydantic_v1 import BaseModel
import os

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: Any


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)


model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

tool = YouTubeSearchTool()
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
agent = create_openai_functions_agent(llm, [tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)