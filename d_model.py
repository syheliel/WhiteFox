from modelscope.hub.snapshot_download import snapshot_download

cache_dir = "./llm_model"
name = "Qwen/Qwen2.5-Coder-32B-Instruct"
name = "Salesforce/codet5p-110m-embedding"
name = "iic/gte_Qwen2-1.5B-instruct"
model_dir = snapshot_download(name, cache_dir = cache_dir)
