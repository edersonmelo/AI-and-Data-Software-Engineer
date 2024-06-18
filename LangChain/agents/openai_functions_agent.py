from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent, load_tools
from langchain_openai import ChatOpenAI
import os

os.environ['OPENAI_API_KEY']='YOUR_API_KEY'

prompt = hub.pull("hwchase17/openai-functions-agent")

llm = ChatOpenAI(model='gpt-4-1106-preview')

tools = load_tools(
    ["arxiv"],
)

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke({"input": "what is this paper 2312.16862v1 about?"}))

