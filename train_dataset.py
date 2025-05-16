from datasets import load_dataset
from transformers import AutoTokenizer
from dataset import VisRAGRetrievalDataset

hf_dataset = load_dataset("openbmb/VisRAG-Ret-Train-Synthetic-data", split="train")
tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = VisRAGRetrievalDataset(hf_dataset, tokenizer)


print(train_dataset[0])
