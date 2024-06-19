from fastapi import FastAPI
import requests
from llama_cpp import Llama 
import threading

app = FastAPI()

llm = None

def start_llm():
    llm = Llama(model_path="./tinyllama-1.1b-chat.gguf")
    
 
@app.post("/health")
    return {"status": "ok"}

@app.post("/llm")
async def stream(item: dict):
    
    if llm is None:
        raise ValueError("modelo carregando, por favor tente mais tarde")
	
    if 'prompt' not in item.keys():
        raise ValueError("prompt é obrigatório")

    prompt = item['prompt']
    temperatura = item['temperatura'] if 'temperatura' in item.keys() else 0.2
    max_tokens = item['max_tokens'] if 'max_tokens' in item.keys() else 512
    
    return llm(prompt, max_tokens=max_tokens, temperature=temperatura)
    
    
threading.Thread(target=start_llm).start()