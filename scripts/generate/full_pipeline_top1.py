import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from PIL import Image
import torch.nn.functional as F
import os

# ==== 配置路径 ====
retriever_path = "/media/mldadmin/home/s124mdg39_04/models/VisRAG-Ret"
generator_path = "/media/mldadmin/home/s124mdg39_04/models/MiniCPM-V-2"
image_dir = "/media/mldadmin/home/s124mdg39_04/VisRAG/images_test"  # 存放图像的文件夹
query = "What does a dog look like?"

# ==== 加载检索器 ====
ret_tokenizer = AutoTokenizer.from_pretrained(retriever_path, trust_remote_code=True)
ret_model = AutoModel.from_pretrained(retriever_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

# ==== 加载生成器 ====
gen_tokenizer = AutoTokenizer.from_pretrained(generator_path, trust_remote_code=True)
gen_model = AutoModelForCausalLM.from_pretrained(generator_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

# ==== 检索 embedding 工具函数 ====
def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    return s / d

@torch.no_grad()
def encode_batch(text_list, image_list):
    inputs = {"text": text_list, "image": image_list, "tokenizer": ret_tokenizer}
    outputs = ret_model(**inputs)
    reps = weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
    return F.normalize(reps, p=2, dim=1)

# ==== 准备图像和query ====
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
images = [Image.open(p).convert("RGB") for p in image_paths]
queries = ["Represent this query for retrieving relevant documents: " + query]

# ==== 编码并计算相似度 ====
query_emb = encode_batch(queries, [None])
doc_emb = encode_batch(["" for _ in images], images)
scores = (query_emb @ doc_emb.T)[0]
top_idx = scores.argmax().item()

# ==== 构造 prompt ====
context_text = f"This is the retrieved image: {os.path.basename(image_paths[top_idx])}"
prompt = f"Question: {query}\nContext: {context_text}\nAnswer:"

# ==== 生成回答 ====
inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)
with torch.no_grad():
    output_ids = gen_model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False
    )
    answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\nQuery:", query)
print("Top-1 Image:", image_paths[top_idx])
print("Answer:", answer)