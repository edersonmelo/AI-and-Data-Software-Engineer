import chainlit as cl
from langchain.agents import AgentType, initialize_agent
from langchain.tools import StructuredTool
from langchain.chat_models import ChatOpenAI
from langchain.tools import YouTubeSearchTool, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
import requests
from pytube import YouTube
from dotenv import load_dotenv
import requests
import os
import json
import base64
import os

load_dotenv()

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY'] = 'LANGCHAIN_API_KEY'
os.environ['LANGCHAIN_PROJECT'] = "LANGCHAIN_PROJECT"


def baixar_video(url):
    
    link = YouTube(url).streams.filter(only_audio=True).first()
    new_file = link.title.replace("[^a-zA-Z]", "")+".mp4"
    if not os.path.exists(new_file):
        audio = link.download()
        base, ext = os.path.splitext(audio)
        filename = os.path.basename(audio)
        new_file = base.replace(filename, filename.replace("[^a-zA-Z]", "")) + '.mp4'
        os.rename(audio, new_file)
    with open(new_file, 'rb') as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
    return audio_base64

def transcrever_audio(url):
    audio = baixar_video(url)
    
    url = "https://api.runpod.ai/v2/faster-whisper/runsync"

    payload = {
        "input": {
            "audio_base64": audio,
            "model": "base",
            "transcription": "plain_text",
            "translate": False,
            "language": "en",
            "temperature": 0,
            "best_of": 5,
            "beam_size": 5,
            "patience": 1,
            "suppress_tokens": "-1",
            "condition_on_previous_text": False,
            "temperature_increment_on_fallback": 0.2,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1,
            "no_speech_threshold": 0.6,
            "word_timestamps": False
        },
        "enable_vad": False
    }
    headers = {
        "authorization": "Bearer "+os.getenv("RUNPOD_KEY"),
        "accept": "application/json",
        "content-type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    return json.loads(response.text)['output']['transcription']
    
    
    
messages = []

llm = ChatOpenAI(model_name="gpt-4-1106-preview")

tools = [
    YouTubeSearchTool(),
    Tool.from_function(
        func=transcrever_audio,
        name="TranscreveAudioVideo",
        description="useful for when you need to transcribe the audio of a youtube video, receives the video url as input",
    )
]

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

agent_executor = initialize_agent(
    tools, llm, agent=AgentType.OPENAI_FUNCTIONS, memory=memory, verbose=True, agent_kwargs=agent_kwargs,
    max_execution_time=100000,
    max_iterations=100
)

@cl.on_message
async def main(message: cl.Message):

    msg = cl.Message(content="")
    await msg.send()
    
    system = "Instructions: Make a smart use of the memory you have from the conversation, if the user asks to work on something already processed before get the results from your memory and start working on the new request from there\n User:"
    
    response = await agent_executor.ainvoke({"input": system+message.content})
    memory.save_context({"input": message.content}, {"output": response['output']})
    msg.content = response["output"]
    
    await msg.update()
    
