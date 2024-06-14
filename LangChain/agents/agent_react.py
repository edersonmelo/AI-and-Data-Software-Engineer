from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain import hub
import os

os.environ['OPENAI_API_KEY']='YOUR_API_KEY'

prompt = hub.pull("hwchase17/react")

llm = OpenAI(model='gpt-3.5-turbo-instruct')

tools = load_tools(
    ["arxiv"],
)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

print(agent_executor.invoke({"input": "what is this paper 2312.16862v1 about?"}))
print(agent_executor.invoke({"input": "Number of TinyGPT paper?"}))
