from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import os

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

messages = [
    SystemMessage(
        content="You are a helpful assistant"
    ),
    HumanMessage(
        content="O que é AGI?"
    ),
]

print(chat(messages))



