from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain
import boto3
import os

os.environ['AWS_ACCESS_KEY_ID'] = "AWS_ACCESS_KEY_ID"
os.environ['AWS_SECRET_ACCESS_KEY'] = "AWS_SECRET_ACCESS_KEY"


BEDROCK_CLIENT = boto3.client("bedrock-runtime", 'us-east-1')
llm = Bedrock(
    client=BEDROCK_CLIENT, model_id="anthropic.claude-v2",
    model_kwargs={"max_tokens_to_sample": 1000}
)

developer_template = """You are a very experient senior developer. \
You are great at answering questions about software development in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""
developer_prompt = PromptTemplate.from_template(developer_template)

qa_template = """You are a very good QA analyst. You are great at answering QA questions. \
You are so good because you are able to list the tests needed to make an avaliation of a solution. \

Here is a question:
{input}"""
qa_prompt = PromptTemplate.from_template(qa_template)

prompt_infos = [
    {
        "name": "developer",
        "description": "Good for answering questions about software development",
        "prompt_template": developer_template,
    },
    {
        "name": "qa",
        "description": "Good for answering about testing software and solutions",
        "prompt_template": qa_template,
    },
]

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
    
default_chain = ConversationChain(llm=llm, output_key="text")

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

print(chain.run("What are best practices while testing solutions?"))