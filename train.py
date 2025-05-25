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

class Retriever_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.visual_proj_modules = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=True),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim, bias=True),
        ])
        self.text_proj_modules = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=True),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim, bias=True),
        ])

    def forward(self, text_embedding, visual_embedding):
        for module in self.visual_proj_modules:
            visual_embedding = module(visual_embedding)
        for module in self.text_proj_modules:
            text_embedding = module(text_embedding)
        return visual_embedding, text_embedding

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
def train(args):
    # 初始化分布式训练
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"model pth save_dir: {args.save_dir}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    hf_dataset = load_dataset("openbmb/VisRAG-Ret-Train-Synthetic-data", split="train")
    dataset = VisRAGRetrievalDataset(hf_dataset, tokenizer)
    
    # 使用DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        collate_fn=collate_fn, 
        num_workers=args.num_workers
    )

    # 构造 DRModel（使用 build 方法）
    model_args = ModelArguments(
        model_name_or_path="/media/mldadmin/home/s124mdg39_04/models/MiniCPM-V-2",
        attention="causal",
        pooling="lasttoken",
        feature="last_hidden_state",
        normalize=True,
        dtype="float32"
    )
    data_args = DataArguments(q_max_len=64, p_max_len=64)
    training_args = DRTrainingArguments(
        per_device_train_batch_size=4,
        output_dir="./checkpoints",
        num_train_epochs=3
    )

    model = DRModel.build(model_args=model_args, data_args=data_args, train_args=training_args, trust_remote_code=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.lm_q.llm.model.layers = None

    retriever_model = Retriever_Model(2304, 2560)
    retriever_model = retriever_model.to(device)
    # 包装模型为DDP模型
    retriever_model = torch.nn.parallel.DistributedDataParallel(
        retriever_model,
        device_ids=[local_rank],
        output_device=local_rank
    )
    retriever_model.train()

    optimizer = torch.optim.AdamW(retriever_model.parameters(), lr=args.lr)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=1000, num_training_steps=len(dataloader) * args.epochs)

    steps = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # 确保每个epoch的数据打乱方式不同
        total_loss = 0.0
        for batch in tqdm(dataloader, disable=local_rank != 0):
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            with torch.no_grad():
                text_embedding = model.lm_q.llm.model.embed_tokens(query_input_ids)
                # 考虑attention mask
                text_embedding = text_embedding * query_attention_mask.unsqueeze(-1)

                visual_embedding = model.lm_q.get_vision_embedding(pixel_values)
            
            visual_embedding, text_embedding = retriever_model(text_embedding, visual_embedding)

            logits = torch.matmul(text_embedding.mean(dim=1), visual_embedding.mean(dim=1).T)
            labels = torch.arange(logits.size(0)).to(device)
            loss = nn.CrossEntropyLoss()(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if steps % args.print_every_steps == 0 and local_rank == 0:
                print(f"steps: {steps}, loss: {loss.item()}")

            steps += 1
            total_loss += loss.item()

            if steps % args.save_every_steps == 0 and local_rank == 0:
                torch.save(retriever_model.module.state_dict(), os.path.join(args.save_dir, f"retriever_model_state_dict_{steps}.pth"))

        if local_rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch}] avg loss: {avg_loss:.4f}")

    # 清理进程组
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=30)
    parser.add_argument("--save_dir", type=str, default="/media/mldadmin/home/s124mdg39_04/VisRAG/checkpoints")
    parser.add_argument("--save_every_steps", type=int, default=2000)
    parser.add_argument("--print_every_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)