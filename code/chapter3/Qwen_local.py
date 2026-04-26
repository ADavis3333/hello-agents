import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置环境变量，避免连接中断
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 指定模型ID
model_id = "Qwen/Qwen1.5-0.5B-Chat"

# 设置设备，优先使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("模型和分词器加载完成！")

# 初始化对话记录
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# 对话循环
while True:
    # 获取用户输入
    user_input = input("用户: ")

    # 如果用户输入"退出"或"exit"，结束对话
    if user_input.lower() in ["退出", "exit"]:
        print("结束对话。")
        break

    # 将用户输入添加到对话记录中
    messages.append({"role": "user", "content": user_input})

    # 使用分词器的模板格式化输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入文本
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 使用模型生成回答
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )

    # 将生成的 Token ID 截取掉输入部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码生成的 Token ID
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 将模型的回答添加到对话记录中
    messages.append({"role": "assistant", "content": response})

    # 输出模型的回答
    print("\n模型: " + response)