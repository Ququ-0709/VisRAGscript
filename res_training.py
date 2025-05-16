# 这个的问题是一直没有产生embedding 所以改了源码的输出
# ===================== Step 0: 导入依赖 =====================
import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from openmatch.modeling import DRModel
from openmatch.modeling.modeling_minicpmv.modeling_minicpmv import LlamaTokenizerWrapper
from types import SimpleNamespace as Namespace
from openmatch.modeling.modeling_visrag_ret import VisRAG_Ret  # ✅ 强制用你改写的类

# ===================== Step 1: 定义 Dataset =====================
class VisRAGRetDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx): #告诉 DataLoader 每条数据怎么处理成张量
        example = self.dataset[idx]
        image = self.transform(example['image'])
        query = example['query']

        tokenized = self.tokenizer(
            query,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            "image": image,
            "input_ids": tokenized['input_ids'].squeeze(0),
            "attention_mask": tokenized['attention_mask'].squeeze(0)
        }
# Dataset 的 __getitem__ 返回的是一条样本（图像 + 文本）
# DataLoader 的作用是：批量加载样本，组成 batch
# ===================== Step 2: 定义损失函数 =====================
def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    logits = torch.matmul(image_embeds, text_embeds.T) / temperature
    labels = torch.arange(image_embeds.size(0)).to(image_embeds.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


# ===================== Step 3: 编写训练主逻辑 =====================
def main():
    model_path = "/media/mldadmin/home/s124mdg39_04/models/VisRAG-Ret"  # TODO: 修改成你的模型路径
    batch_size = 16
    max_length = 128
    num_epochs = 3
    lr = 1e-5

    # 1. 加载数据集
    dataset = load_dataset("openbmb/VisRAG-Ret-Train-Synthetic-data", split="train[:1000]")

    # 2. 加载 tokenizer
    tokenizer = LlamaTokenizerWrapper.from_pretrained(model_path)

    # 3. 初始化模型
    from types import SimpleNamespace as Namespace

    model_args = Namespace(
        model_name_or_path=model_path,
        tokenizer_name=None,
        processor_name=None,
        cache_dir=None,

        attention="causal",         # 或 "default" 或 "bidirectional"，取决于你配置
        pooling="cls",              # "cls" 是 MiniCPM 通常使用的 pooling 方式
        untie_encoder=False,
        feature="last_hidden_state",

        add_linear_head=False,
        projection_in_dim=768,
        projection_out_dim=768,

        dtype="float16",            # 根据你的模型设置 float16 / bfloat16 / float32
        encoder_only=False,

        normalize=True,
        lora=False,
        lora_r=32,

        attn_implementation="sdpa",  # 推荐用 'sdpa'，兼容性强
    )
   

    '''model = DRModel.build(
        model_args=model_args,
        data_args=None,
        train_args=None,
        cache_dir=None
    )'''
    # ✅ 强制加载你改写的 VisRAG_Ret 类（不经过 AutoModel）
    model = VisRAG_Ret.from_pretrained(model_path)
    model.train().cuda()

    # 4. 创建 Dataset + Dataloader
    train_dataset = VisRAGRetDataset(dataset, tokenizer, max_length)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 你现在的 DataLoader 能工作是因为 Dataset 返回的结构统一、shape 固定，
# default collate 就能把它拼成一个 batch tensor。不需要collate.fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 5. 开始训练
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].cuda()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()

            '''outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=images
            )'''

            # 第一次：处理文本
            q_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mode="query"
            )
            

            # 第二次：处理图像
            p_out = model(
                image=images,
                mode="passage"
            )
            print("q_out:", q_out)
            print("p_out:", p_out)

            text_embeds = q_out.q_reps  # B x D
            image_embeds = p_out.p_reps  # B x D
 
            # image_embeds = outputs["p_reps"]  # 图像（p = passage）
            # text_embeds = outputs["q_reps"]  # 文本（q = query）

            loss = contrastive_loss(image_embeds, text_embeds)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss / len(dataloader):.4f}")

    # 6. 保存模型
    torch.save(model.state_dict(), "visrag_ret_model.pt")


if __name__ == "__main__":
    main()
