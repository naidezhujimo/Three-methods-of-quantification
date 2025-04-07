import torch

def absmax_quantize(X):
    scale = 127 / torch.max(torch.abs(X)) # 计算缩放因子(防止溢出)
    X_quant = (scale * X).round()        # 缩放后取整得到量化值
    X_dequant = X_quant / scale          # 反量化恢复近似值
    return X_quant.to(torch.int8), X_dequant 

def zeropoint_quantize(X):
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range  # 处理全零张量
    scale = 255 / x_range  # 缩放因子覆盖整个int8范围
    zeropoint = (-scale * torch.min(X) - 128).round()  # 零点偏移计算
    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)  # 量化+截断
    X_dequant = (X_quant - zeropoint) / scale  # 反量化
    return X_quant.to(torch.int8), X_dequant


#----------------------------------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载gpt-2模型和分词器
model_id = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 模型参数大小
print(f"Model size: {model.get_memory_footprint():,} bytes")

# 权重样例输出
weights = model.transformer.h[0].attn.c_attn.weight.data
print("Original weights:")
print(weights)

weights_abs_quant, _ = absmax_quantize(weights)
print("\nAbsmax quantized weights:")
print(weights_abs_quant)

weights_zp_quant, _ = zeropoint_quantize(weights)
print("\nZero-point quantized weights:")
print(weights_zp_quant)

#-------------------------------------------模型量化----------------------------------------------

import numpy as np
from copy import deepcopy

# 保存原始权重
weights = [param.data.clone() for param in model.parameters()]

# 创建absmax量化模型
model_abs = deepcopy(model)
weights_abs = []
for param in model_abs.parameters():
    _, dequantized = absmax_quantize(param.data)
    param.data = dequantized
    weights_abs.append(dequantized)

# 创建zeropoint量化模型
model_zp = deepcopy(model)
weights_zp = []
for param in model_zp.parameters():
    _, dequantized = zeropoint_quantize(param.data)
    param.data = dequantized
    weights_zp.append(dequantized)

# 8位量化
model_int8 = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map='auto',
                                             load_in_8bit=True,
                                             )


# -------------------------------------- 权重分布可视化 --------------------------------------
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 展开权重张量
weights = np.concatenate([t.cpu().numpy().flatten() for t in weights])
weights_abs = np.concatenate([t.cpu().numpy().flatten() for t in weights_abs])
weights_zp = np.concatenate([t.cpu().numpy().flatten() for t in weights_zp])
weights_int8 = [param.data.clone() for param in model_int8.parameters()]
weights_int8 = np.concatenate([t.cpu().numpy().flatten() for t in weights_int8])
# 设置背景样式
plt.style.use('ggplot')

# 创建图表和子图
fig, axs = plt.subplots(2, figsize=(8,7), dpi=150, sharex=True)

# 第一个子图：原始权重 vs absmax 量化权重
axs[0].hist(weights, bins=150, alpha=0.5, label='Original weights', color='blue', range=(-2, 2))
axs[0].hist(weights_abs, bins=150, alpha=0.5, label='Absmax weights', color='red', range=(-2, 2))

# 第二个子图：原始权重 vs zeropoint 量化权重
axs[1].hist(weights, bins=150, alpha=0.5, label='Original weights', color='blue', range=(-2, 2))
axs[1].hist(weights_zp, bins=150, alpha=0.5, label='Zero-point weights', color='green', range=(-2, 2))

# 添加网格
for ax in axs:
    ax.grid(True, linestyle='--', alpha=0.6)

# 添加图例
axs[0].legend()
axs[1].legend()

# 添加标题和标签
axs[0].set_title('Comparison of Original and Absmax Quantized Weights', fontsize=16)
axs[1].set_title('Comparison of Original and Zeropoint Quantized Weights', fontsize=16)

for ax in axs:
    ax.set_xlabel('Weights', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

# 调整字体和布局
plt.rc('font', size=12)

plt.tight_layout()
plt.show()


# 原始权重  vs  8位量化权重
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8,7), dpi=150)

ax.hist(weights, bins=150, alpha=0.5, label='Original weights',
        color='blue', range=(-2, 2))
ax.hist(weights_int8, bins=150, alpha=0.5, label='LLM.int8() weights',
        color='red', range=(-2, 2))

ax.grid(True, linestyle='--', alpha=0.6)

ax.legend()

ax.set_title('Comparison of Original and Dequantized Weights', fontsize=16)
ax.set_xlabel('Weights', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.gca().yaxis.set_major_formatter(ticker.EngFormatter())

plt.rc('font', size=12)

plt.tight_layout()
plt.show()

#--------------------------------------文本生成与困惑度评估-----------------------------------------------

# 文本生成函数
def generate_text(model, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device) #编码(返回pytorch的tensor)
    output = model.generate(inputs=input_ids,  #输入 token IDs
                            max_length=max_length,  #生成文本的最大长度
                            do_sample=True,  #启用采样模式（而不是贪婪解码）
                            top_k=30,  #限制采样时只从概率最高的 30 个 token 中选择
                            pad_token_id=tokenizer.eos_token_id,    #设置填充 token 为结束 token
                            attention_mask=input_ids.new_ones(input_ids.shape))  #创建注意力掩码，确保模型只关注有效 token
    return tokenizer.decode(output[0], skip_special_tokens=True) #解码


original_text = generate_text(model, "I want to play")
absmax_text   = generate_text(model_abs, "I want to play")
zp_text       = generate_text(model_zp, "I want to play")
text_int8 = generate_text(model_int8, "I want to play")

print("-" * 50)
print(f"Original model:\n{original_text}")
print("-" * 50)
print(f"Absmax model:\n{absmax_text}")
print("-" * 50)
print(f"Zeropoint model:\n{zp_text}")
print("-" * 50)
print(f"LLM.int8() model:\n{text_int8}")


# 计算模型困惑度
def calculate_perplexity(model, text):

    encodings = tokenizer(text, return_tensors='pt').to(device)

    input_ids = encodings.input_ids
    target_ids = input_ids.clone()

    with torch.no_grad(): #禁用梯度计算，减少内存占用
        outputs = model(input_ids, labels=target_ids)

    # 计算负对数似然损失
    neg_log_likelihood = outputs.loss 

    # 计算困惑度(困惑度是负对数似然损失的指数值)
    ppl = torch.exp(neg_log_likelihood) 

    return ppl

ppl     = calculate_perplexity(model, original_text)
ppl_abs = calculate_perplexity(model_abs, absmax_text)
ppl_zp  = calculate_perplexity(model_zp, absmax_text)
ppl_int8 = calculate_perplexity(model_int8, text_int8)

print("-" * 50)
print(f"Original perplexity:  {ppl.item():.2f}")
print(f"Absmax perplexity:    {ppl_abs.item():.2f}")
print(f"Zeropoint perplexity: {ppl_zp.item():.2f}")
print(f"Perplexity (LLM.int8()): {ppl_int8.item():.2f}")



print("\n==== 模型压缩率 ====")
print(f"原始模型: {model.get_memory_footprint():,} bytes")
print(f"8位量化后Model size: {model_int8.get_memory_footprint():,} bytes")
print(f"量化后模型大小缩减比例：{100 * ((model.get_memory_footprint() - model_int8.get_memory_footprint()) / model.get_memory_footprint()):.2f}%")
