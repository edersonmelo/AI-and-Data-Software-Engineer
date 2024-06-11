from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "sk-proj-OPENAI_API_KEY"


loader = PyPDFLoader("data/2203.15556.pdf")
pages = loader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(pages)

embedding_function = OpenAIEmbeddings()

db = Chroma.from_documents(docs, embedding_function)

query = "Quais benchmarks de senso comum foram usados?"
docs = db.similarity_search(query)

	
chat = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            streaming=True,
        )

print(chat.predict("Considerando somente o contexto abaixo, responda a pergunta de forma simples e fácil de entender para alguém que n]ao seja da área, responda em português brasileiro. \nPergnta: "+query+ "\nContexto:"+str(docs)))