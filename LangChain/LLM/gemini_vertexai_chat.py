from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
import io
import base64
import os

os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "YOUR_CREDENTIALS_FILE.json"

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

image = Image.open("C:\AI-and-Data-Software-Engineer\LangChain\LLM\imagem.png")
# Create a bytes buffer
buffer = io.BytesIO()

# Save image to the buffer
image.save(buffer, format="PNG")

# Get the buffer's content as a byte string
buffered_png = buffer.getvalue()

# Encode the byte string in Base64 and decode it to a UTF-8 string
base64_encoded_string = base64.b64encode(buffered_png).decode("utf-8")

# Format the Base64 string
data_url = f"data:image/png;base64,{base64_encoded_string}"

llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
# example
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
    ]
)
print(llm.invoke([message]))