from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, TransformChain
from langchain.llms import Bedrock
import re
import boto3
import os

os.environ['AWS_ACCESS_KEY_ID'] = "AWS_ACCESS_KEY_ID"
os.environ['AWS_SECRET_ACCESS_KEY'] = "AWS_SECRET_ACCESS_KEY"


BEDROCK_CLIENT = boto3.client("bedrock-runtime", 'us-east-1')
llm = Bedrock(
    client=BEDROCK_CLIENT, model_id="anthropic.claude-v2",
    model_kwargs={"max_tokens_to_sample": 1000}
)

prompt = PromptTemplate.from_template(
    """Summarize this text:

{output_text}

Summary:"""
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    padrao = r"\[(\d+)\]"
    texto_sem_numeros = re.sub(padrao, "", text)
    return {"output_text": texto_sem_numeros}
    

transform_chain = TransformChain(
    input_variables=["text"], output_variables=["output_text"], transform=transform_func
)

sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])

texto = ""
with open("texto.txt", "r", encoding="utf8") as f:
    texto = f.read()
    
print(sequential_chain.run(texto))