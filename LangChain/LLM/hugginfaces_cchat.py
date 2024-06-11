from langchain.llms.huggingface_pipeline import HuggingFacePipeline

hf = HuggingFacePipeline.from_model_id(
    model_id="MBZUAI/LaMini-Neo-125M",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 500},
)

print(hf("Em IA o que é AGI?"))
