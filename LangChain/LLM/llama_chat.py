from langchain.llms import LlamaCpp

prompt = """
<|im_start|>system
You are a helpfull assistant<|im_end|>
<|im_start|>user
Em AI o que é AGI?<|im_end|>
<|im_start|>assistant
"""

llm = LlamaCpp(
    model_path="C:\\AI-and-Data-Software-Engineer\\LangChain\\LLM\\llama\\models\\orca-2-7b.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=2000
)

print(llm(prompt))
