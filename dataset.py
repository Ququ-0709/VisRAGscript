import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

class VisRAGRetrievalDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=64):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        query = sample["query"]

        pixel_values = self.image_transform(image)

        query_enc = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "query_input_ids": query_enc["input_ids"][0],
            "query_attention_mask": query_enc["attention_mask"][0],
            "pixel_values": pixel_values
        }
