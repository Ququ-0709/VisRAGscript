from transformers import AutoModel, AutoTokenizer
from modelscope import AutoModel as CPMModel, AutoTokenizer as CPMTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import os
import time
import numpy as np
import peft


retriever_model_path = "/media/mldadmin/home/s124mdg39_04/models/VisRAG-Ret"
generator_model_path = "/media/mldadmin/home/s124mdg39_04/models/MiniCPM-V-2"
image_folder = "/media/mldadmin/home/s124mdg39_04/VisRAG/images_test"        # ✅ [你放图片的本地目录]

def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

@torch.no_grad()
def encode(text_or_image_list):
    
    print("🧭 Encoding input...")
    start = time.time()

    if (isinstance(text_or_image_list[0], str)):
        inputs = {
            "text": text_or_image_list,
            'image': [None] * len(text_or_image_list),
            'tokenizer': tokenizer
        }
    else:
        inputs = {
            "text": [''] * len(text_or_image_list),
            'image': text_or_image_list,
            'tokenizer': tokenizer
        }
    outputs = model(**inputs)
    print(f" encode done in {time.time() - start:.2f}s")
    attention_mask = outputs.attention_mask
    hidden = outputs.last_hidden_state

    reps = weighted_mean_pooling(hidden, attention_mask)   
    #每个样本的 embedding 做 L2 归一化（单位向量化），方便后续做 相似度计算 或 对比学习
    embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
    return embeddings

# model_name_or_path = "openbmb/VisRAG-Ret"
tokenizer = AutoTokenizer.from_pretrained(retriever_model_path, trust_remote_code=True)
#model = AutoModel.from_pretrained(retriever_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = AutoModel.from_pretrained(retriever_model_path, trust_remote_code=True)
model = model.half().cuda()
model.eval()

# 构造输入 
script_dir = os.path.dirname(os.path.realpath(__file__))
query = "what does this dog look like?"
INSTRUCTION = "Represent this query for retrieving relevant documents: "
queries = [INSTRUCTION + query]
passages = [
    Image.open(os.path.join(script_dir, image_folder, 'cat.png')).convert('RGB'),
    Image.open(os.path.join(script_dir, image_folder, 'dog.png')).convert('RGB'),
    Image.open(os.path.join(script_dir, image_folder, 'elephant.png')).convert('RGB'),
]



# -------------------- 检索阶段 --------------------
embeddings_query = encode(query)
embeddings_doc = encode(passages)

print("embeddings_query.shape =", embeddings_query.shape)
print("embeddings_doc.shape =", embeddings_doc.shape)

scores = (embeddings_query @ embeddings_doc.T)  # shape: [num_docs]
# scores[i][j] = 第 i 个 query 和第 j 张图 的相似度分数
print("scores shape =", scores.shape)
topk_indices = torch.topk(torch.tensor(scores), k=3).indices.tolist()

print(f"Top-3 matches: {topk_indices}")
for i in range(len(topk_indices)):
    for j in topk_indices[i]:
       
        score_val = scores[i][j]
        if isinstance(score_val, np.ndarray):
            score_val = score_val.squeeze().item()
        else:
            score_val = float(score_val)
        print(f"Score: {score_val:.4f} - Query {i} - Image {j}")


cpm_model = CPMModel.from_pretrained(generator_model_path, trust_remote_code=True).half().cuda()
cpm_tokenizer = CPMTokenizer.from_pretrained(generator_model_path, trust_remote_code=True)

# -------------------- 构造 LLM 输入并生成回答 --------------------
top_image = passages[topk_indices[0][0]]
question = query

print("\nRunning MiniCPM-V-2 inference...")
print(" MiniCPM model loaded.")
print(" Starting generation...")

msgs = [{"role": "user", "content": question}]  # 构造最基本的对话轮
response = cpm_model.chat(
    image=top_image,
    msgs=msgs,
    context=None,
    tokenizer=cpm_tokenizer,
    max_new_tokens=64
)
# response = cpm_model.chat(cpm_tokenizer, query=question, image=top_image, max_new_tokens=64)
print(" Generation finished.")
print("\nGenerated Answer:\n", response)

