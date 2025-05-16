import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "src"))

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm import tqdm

from openmatch.modeling import DRModel
from openmatch.arguments import ModelArguments, DataArguments, DRTrainingArguments

# ==================== 1. 自定义 Dataset ====================
class VisRAGRetrievalDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=64):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB").resize((224, 224))
        image_tensor = torch.tensor(image.getdata()).float().view(3, 224, 224) / 255.0

        query = sample["query"]
        query_enc = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "pixel_values": image_tensor
        }

# ==================== 2. Collate 函数 ====================
def collate_fn(batch):
    return {
        "query_input_ids": torch.stack([x["query_input_ids"] for x in batch]),
        "query_attention_mask": torch.stack([x["query_attention_mask"] for x in batch]),
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
    }

# ==================== 3. 主训练逻辑 ====================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    hf_dataset = load_dataset("openbmb/VisRAG-Ret-Train-Synthetic-data", split="train")
    dataset = VisRAGRetrievalDataset(hf_dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 构造 DRModel（使用 build 方法）
    model_args = ModelArguments(
        model_name_or_path="/media/mldadmin/home/s124mdg39_04/models/MiniCPM-V-2",
        attention="causal",
        pooling="lasttoken",
        feature="last_hidden_state",
        normalize=True,
        dtype="float16"
    )
    data_args = DataArguments(q_max_len=64, p_max_len=64)
    training_args = DRTrainingArguments(
        per_device_train_batch_size=4,
        output_dir="./checkpoints",
        num_train_epochs=3
    )

    model = DRModel.build(model_args=model_args, data_args=data_args, train_args=training_args)
    model.to(device)
    model.train()

    # 优化器调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 3)

    for epoch in range(3):
        total_loss = 0.0
        for batch in tqdm(dataloader):
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            query_inputs = {"input_ids": query_input_ids, "attention_mask": query_attention_mask}
            passage_inputs = {"pixel_values": pixel_values}

            outputs = model(query=query_inputs, passage=passage_inputs)
            q_reps, p_reps = outputs.q_reps, outputs.p_reps

            logits = torch.matmul(q_reps, p_reps.T)
            labels = torch.arange(logits.size(0)).to(device)
            loss = nn.CrossEntropyLoss()(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}] avg loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()