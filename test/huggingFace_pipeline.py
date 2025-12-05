import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 本地模型路径
local_model_path = "/Users/yp/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto")

# 创建 Pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# 关键：TinyLlama 的 Chat 模板
prompt = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "你好，"},
    ],
    tokenize=False,
    add_generation_prompt=True
)

# 推理
output = generator(
    prompt,
    max_new_tokens=150,
    do_sample=False  # 更稳定输出
)

print("=== 生成结果 ===")
print(output[0]["generated_text"])
print(output)

