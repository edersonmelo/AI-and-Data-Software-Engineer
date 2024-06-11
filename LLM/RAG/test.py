from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "sk-OPENAI_API_KEY"


try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
except:
    from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("./data").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()
	
	
chat = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            streaming=True,
        )
        
pergunta = "Given a fixed FLOPs budget, how should one trade-off model size and the number of training tokens?"       
query_engine = index.as_query_engine()
response = query_engine.query(pergunta)
print(response)

print(chat.predict("Considerando somente o contexto abaixo, responda a pergunta de forma simples e fácil de entender para alguém que n]ao seja da área, responda em português brasileiro. \nPergnta: "+pergunta+ "\nContexto:"+str(response)))