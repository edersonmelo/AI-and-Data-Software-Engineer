from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage
import os
import boto3

os.environ['AWS_ACCESS_KEY_ID'] = "AWS_ACCESS_KEY_ID"
os.environ['AWS_SECRET_ACCESS_KEY'] = "AWS_SECRET_ACCESS_KEY"

BEDROCK_CLIENT = boto3.client("bedrock-runtime", 'us-east-1')

chat = BedrockChat(model_id="meta.llama2-13b-chat-v1", model_kwargs={"temperature": 0.1}, client=BEDROCK_CLIENT)

messages = [
    HumanMessage(
        content="Make clothing style suggestions for a hot day"
    )
]
print(chat(messages))