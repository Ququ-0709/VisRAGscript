from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from modelscope import AutoModel as CPMModel, AutoTokenizer as CPMTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from difflib import SequenceMatcher

# 创建一个空白图像作为 image 占位
#dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))

# ========== Step 1: 加载 DocVQA 示例数据集 ==========
dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")
# dataset = dataset.select(range(200))  # ⬅️ 只保留前 200 条样本
# ========== Step 2: 配置检索器模型 (bge-large) ==========
# retriever_model_path = "/media/mldadmin/home/s124mdg39_04/models/bge-large-en"
retriever_model_path = "/media/mldadmin/home/s124mdg39_04/models/VisRAG-Ret"
retriever_model = AutoModel.from_pretrained(retriever_model_path, trust_remote_code=True).eval().cuda()
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_path, trust_remote_code=True)


# retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_path)
# retriever_model = AutoModel.from_pretrained(retriever_model_path).eval().cuda()

# ========== Step 3: 配置生成器模型 (MiniCPM-V 2.6) ==========
generator_model_path = "/media/mldadmin/home/s124mdg39_04/models/MiniCPM-V-2"
generator_model = CPMModel.from_pretrained(generator_model_path, trust_remote_code=True).half().cuda()
generator_tokenizer = CPMTokenizer.from_pretrained(generator_model_path, trust_remote_code=True)

'''# ========== Step 4: 编码函数（bge embedding） ==========
@torch.no_grad()
def encode_texts(text_list, tokenizer, model, max_length=512):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    embeddings = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)  # 取 CLS 向量
    return embeddings.cpu()'''

# ========== Step 4: 准备图像和查询对 ==========
def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    return s / d

@torch.no_grad()
def encode_batch(text_or_image_list):
    if isinstance(text_or_image_list[0], str):
        inputs = {"text": text_or_image_list, "image": [None] * len(text_or_image_list), "tokenizer": retriever_tokenizer}
    else:
        inputs = {"text": [""] * len(text_or_image_list), "image": text_or_image_list, "tokenizer": retriever_tokenizer}
    outputs = retriever_model(**inputs)
    reps = weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
    return F.normalize(reps, p=2, dim=1).cpu()

query_list = []
image_list = []
doc_ids = []
for example in dataset:
    if example["image"] is not None and "en" in example["query"]:
        doc_ids.append(example["id"])
        image_list.append(example["image"])
        query_list.append((example["query"]["en"], example["id"], example["answers"][0] if example["answers"] else ""))

'''# ========== Step 5: 构造 OCR 文本 Corpus ==========
doc_ids = []
doc_texts = []
query_list = []
answer_list = []

for example in dataset:
    if "en" in example["query"] and example["image"] is not None:
        doc_id = example["id"].split("_")[0]
        ocr_text = " ".join(example["words"])
        query = example["query"]["en"]
        answer = example["answers"][0] if example["answers"] else ""

        doc_ids.append(doc_id)
        doc_texts.append(ocr_text)
        query_list.append((query, doc_id, answer))'''
# 确保图像是 RGB 格式
from PIL import Image
import torch

def ensure_rgb(image):
    if isinstance(image, Image.Image):  # 如果是 PIL 图像
        if image.mode != "RGB":
            image = image.convert("RGB")
    elif isinstance(image, torch.Tensor):  # 如果是张量
        if image.shape[0] == 1:  # 单通道
            image = image.repeat(3, 1, 1)  # 重复三次，变为三通道
    return image
# ========== Step 5: 编码 Corpus 图像 ==========
image_list = [ensure_rgb(img) for img in image_list] 
def encode_images_in_batches(images, batch_size=8):  # ✅ 可根据显存情况调整 batch_size（比如 4、8）
    all_embeddings = []
    for i in tqdm(range(0, len(images), batch_size), desc="Encoding image corpus"):
        batch = images[i:i + batch_size]
        batch_emb = encode_batch(batch)
        all_embeddings.append(batch_emb)
    return torch.cat(all_embeddings, dim=0)

corpus_embeddings = encode_images_in_batches(image_list, batch_size=4)  # 你也可以试 batch_size=4


'''# ========== Step 6: 编码 OCR Corpus ==========
batch_size = 8  # 根据你的 GPU 情况可调节为 4~16
doc_embeddings_list = []

for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding corpus"):
    batch = doc_texts[i:i + batch_size]
    emb = encode_texts(batch, retriever_tokenizer, retriever_model)
    doc_embeddings_list.append(emb)

doc_embeddings = torch.cat(doc_embeddings_list, dim=0)'''


# ========== Step 6: 遍历 Query 执行检索 + 生成 + 评估 ==========
correct_retrieval = 0
correct_generation = 0
total = len(query_list)

for q_idx, (query, gt_doc_id, answer) in enumerate(tqdm(query_list, desc="Evaluating")):
    query_embedding = encode_batch([query])
    scores = torch.matmul(query_embedding, corpus_embeddings.T).squeeze(0)
    top_doc_idx = torch.argmax(scores).item()
    retrieved_doc_id = doc_ids[top_doc_idx]
    retrieved_image = image_list[top_doc_idx]

    # 确保 retrieved_image 是 RGB 格式
    retrieved_image = ensure_rgb(retrieved_image)

    is_correct_retrieval = (retrieved_doc_id == gt_doc_id)
    if is_correct_retrieval:
        correct_retrieval += 1

        msgs = [{
            "role": "user",
            "content": f"You are an expert in document understanding. Please answer the following question: {query}"
        }]

        response = generator_model.chat(
            image=retrieved_image,
            msgs=msgs,
            context=None,
            tokenizer=generator_tokenizer,
            max_new_tokens=64
        )

        def is_match(ans, pred):
            ratio = SequenceMatcher(None, ans.lower().strip(), pred.lower()).ratio()
            return ratio > 0.6

        if isinstance(response, tuple):
            response = response[0]

        if is_match(answer, response):
            correct_generation += 1
'''# ========== Step 7: 遍历 Query 执行检索 + 生成 + 评估 ==========
correct_retrieval = 0
correct_generation = 0
total = len(query_list)

for q_idx, (query, gt_doc_id, answer) in enumerate(tqdm(query_list, desc="Evaluating")):
    query_embedding = encode_texts([query], retriever_tokenizer, retriever_model)
    scores = torch.matmul(query_embedding, doc_embeddings.T).squeeze(0)
    top_doc_idx = torch.argmax(scores).item()
    retrieved_doc_id = doc_ids[top_doc_idx]
    retrieved_text = doc_texts[top_doc_idx]
    retrieved_text = retrieved_text[:500]  # 或者用 NLTK 分句后截前3句

    # 评估1：检索是否命中
    is_correct_retrieval = (retrieved_doc_id == gt_doc_id)
    if is_correct_retrieval:
        correct_retrieval += 1

        # 执行生成器（只对命中样本）
        msgs = [{
        "role": "user",
        "content": f"You are an expert in document understanding. Based on the following OCR text, please answer the question accurately.\n\nOCR:\n{retrieved_text}\n\nQuestion: {query}\nAnswer:"
        }]


        response = generator_model.chat(
            image=dummy_image,  # 占位图像
            msgs=msgs,
            tokenizer=generator_tokenizer,
            context={"text": retrieved_text},  # 真实内容放这里
            max_new_tokens=64
        )
        def is_match(ans, pred):
            ratio = SequenceMatcher(None, ans.lower().strip(), pred.lower()).ratio()
            return ratio > 0.6  # 你可以从 0.6 慢慢调高

        # ⬇️ 防止返回 tuple
        if isinstance(response, tuple):
            response = response[0]

        # ⬇️ 模糊匹配判断
        if is_match(answer, response):
            correct_generation += 1'''


# ========== Step 8: 输出三项指标 ==========
retrieval_acc = correct_retrieval / total
generation_acc = correct_generation / correct_retrieval if correct_retrieval > 0 else 0.0
overall_acc = retrieval_acc * generation_acc

# print("\n📊 Text-RAG 端到端评估结果:")
print("\n📊 VisRAG 端到端评估结果:")
print(f"Correct Retrieval = {retrieval_acc:.2%}")
print(f"Correct Generation = {generation_acc:.2%}")
print(f"Overall Accuracy = {overall_acc:.2%}")
