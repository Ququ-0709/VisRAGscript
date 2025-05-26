from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# ========== Step 1: 加载 HuggingFace 上的 infovqa 子集 ==========
dataset = load_dataset("vidore/infovqa_test_subsampled", split="test")  # 500 个样本

# ========== Step 2: 配置模型路径 ==========
retriever_model_path = "/media/mldadmin/home/s124mdg39_04/models/VisRAG-Ret"

# ========== Step 3: 加载检索模型 ==========
tokenizer = AutoTokenizer.from_pretrained(retriever_model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(retriever_model_path, trust_remote_code=True).half().cuda()
model.eval()

# ========== Step 4: 定义嵌入提取函数 ==========
def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

@torch.no_grad()
def encode_batch(texts=None, images=None):
    if texts:
        inputs = {
            "text": texts,
            "image": [None] * len(texts),
            "tokenizer": tokenizer
        }
    elif images:
        inputs = {
            "text": [''] * len(images),
            "image": images,
            "tokenizer": tokenizer
        }
    else:
        raise ValueError("必须提供 texts 或 images")

    outputs = model(**inputs)
    attention_mask = outputs.attention_mask
    hidden = outputs.last_hidden_state
    reps = weighted_mean_pooling(hidden, attention_mask)
    embeddings = F.normalize(reps, p=2, dim=1).detach().cpu()
    return embeddings

# ========== Step 5: 准备 Corpus 图像集 ==========
corpus_image_ids = []
corpus_images = []

unique_ids = set()
for item in dataset:
    image_id = item["image_filename"]
    if image_id not in unique_ids:
        unique_ids.add(image_id)
        corpus_image_ids.append(image_id)
        corpus_images.append(item["image"])
# 导致OOC
# corpus_embeddings = encode_batch(images=corpus_images)

batch_size = 8
corpus_embeddings_list = []

for i in tqdm(range(0, len(corpus_images), batch_size), desc="Encoding corpus"):
    batch = corpus_images[i:i+batch_size]
    emb = encode_batch(images=batch)
    corpus_embeddings_list.append(emb)

corpus_embeddings = torch.cat(corpus_embeddings_list, dim=0)

# ========== Step 6: 编码 Queries ==========
query_texts = ["Represent this query for retrieving relevant documents: " + item["query"] for item in dataset]
gt_image_ids = [item["image_filename"] for item in dataset]

query_embeddings = encode_batch(texts=query_texts)

# ========== Step 7: 检索并评估 ==========
scores = torch.matmul(query_embeddings, corpus_embeddings.T)  # [Q, N]
topk = 10
topk_indices = scores.topk(k=topk, dim=1).indices

recall, mrr = 0, 0
Q = len(dataset)

for i in range(Q):
    gt_id = gt_image_ids[i]
    retrieved_ids = [corpus_image_ids[j] for j in topk_indices[i]]

    if gt_id in retrieved_ids:
        recall += 1
        rank = retrieved_ids.index(gt_id)
        mrr += 1.0 / (rank + 1)

recall_at_10 = recall / Q
mrr_at_10 = mrr / Q

print(f"\n📊 Evaluation Results on InfoVQA (vidore/infovqa_test_subsampled):")
print(f"Recall@10 = {recall_at_10 * 100:.2f}%")
print(f"MRR@10 = {mrr_at_10:.4f}")
