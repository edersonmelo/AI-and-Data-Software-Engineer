from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
from langchain.chains import LLMChain

import boto3
import os

os.environ['AWS_ACCESS_KEY_ID'] = "AWS_ACCESS_KEY_ID"
os.environ['AWS_SECRET_ACCESS_KEY'] = "AWS_SECRET_ACCESS_KEY"

prompt_template = PromptTemplate.from_template(
    "Translate from {from_lang} to {to_lang} this phrase: {input}"
)

BEDROCK_CLIENT = boto3.client("bedrock-runtime", 'us-east-1')
llm = Bedrock(
    client=BEDROCK_CLIENT, model_id="anthropic.claude-v2",
    model_kwargs={"max_tokens_to_sample": 1000}
)

print("## AI TRANSLATION: ##")
phrase = input("Input to translate: ")
from_lang = input("From: ")
to_lang = input("To: ")

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

print(llm_chain.predict(from_lang=from_lang, input=phrase, to_lang=to_lang))



