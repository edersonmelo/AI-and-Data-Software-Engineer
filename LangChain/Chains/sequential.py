from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.llms import Bedrock
import boto3
import os

os.environ['AWS_ACCESS_KEY_ID'] = "AWS_ACCESS_KEY_ID"
os.environ['AWS_SECRET_ACCESS_KEY'] = "AWS_SECRET_ACCESS_KEY"


BEDROCK_CLIENT = boto3.client("bedrock-runtime", 'us-east-1')
llm = Bedrock(
    client=BEDROCK_CLIENT, model_id="anthropic.claude-v2",
    model_kwargs={"max_tokens_to_sample": 1000}
)


synopsis_prompt = PromptTemplate.from_template(
    """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
)

writer_prompt = PromptTemplate.from_template(
    """You are a experient play writer. Given the synopsis of play, it is your job to write a the script for it.

Play Synopsis:
{synopsis}
Script of the above play:"""
)

synopsis_chain  = LLMChain(llm=llm, prompt=synopsis_prompt)
writer_chain    = LLMChain(llm=llm, prompt=writer_prompt)

overall_chain = SimpleSequentialChain(
    chains=[synopsis_chain, writer_chain], verbose=True
)

print(overall_chain.run("Zombie invade New York"))