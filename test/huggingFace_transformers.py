import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
local_model_path = "/Users/yp/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto")

prompt = "你好，介绍一下哈佛大学。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(inputs)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)
print(output)
print(tokenizer.decode(output[0], skip_special_tokens=False))
# | 推理框架                                  | 特点                           |
# | ------------------------------------- | ---------------------------- |
# | **HuggingFace Transformers（你现在用的方式）** | 最通用，支持所有模型，手动控制最灵活           |
# | **HuggingFace Pipeline**              | 简单易用，高层封装，适合快速实验             |
# | **vLLM**                              | 极快推理，强并发、PagedAttention，适合部署 |
# | **llama.cpp / ggml / gguf**           | 在 CPU / 手机 / 边缘设备上运行量化模型     |
# | **TensorRT-LLM**                      | NVIDIA GPU 上最快推理，适合生产环境      |
# | **OpenVINO**                          | Intel CPU / iGPU 加速          |
# | **ONNX Runtime (ORT)**                | ONNX 模型通用推理引擎，跨设备            |
# | **DeepSpeed-Inference**               | 大模型推理优化（GPU）                 |
# | **FasterTransformer**                 | NVIDIA 官方优化库（TRT-LLM 的前身）    |