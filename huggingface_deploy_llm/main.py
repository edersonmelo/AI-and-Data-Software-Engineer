from fastapi import FastAPI
import requests
from llama_cpp import Llama 
import threading

app = FastAPI()

llm = None

def start_llm():
    global llm  # Adicione esta linha para modificar a variável global
    llm = Llama(model_path="./tinyllama-1.1b-chat.gguf")
    
@app.post("/health")
def health_check():
    return {"status": "ok"}

@app.post("/deployllm")
async def stream(item: dict):
    
    if llm is None:
        raise ValueError("modelo carregando, por favor tente mais tarde")
	
    if 'prompt' not in item.keys():
        raise ValueError("prompt é obrigatório")

    prompt = "<|system|>You are a helpfull assistant</s><|user|>"+item['prompt']+"</s><|assistant|>"
    temperatura = item['temperatura'] if 'temperatura' in item.keys() else 0.2
    max_tokens = item['max_tokens'] if 'max_tokens' in item.keys() else 512
    
    return llm(prompt, max_tokens=max_tokens, temperature=temperatura)
    
threading.Thread(target=start_llm).start()
