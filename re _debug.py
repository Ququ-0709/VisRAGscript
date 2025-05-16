from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import os
import time

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

    print(f"✅ encode done in {time.time() - start:.2f}s")

    attention_mask = outputs.attention_mask
    hidden = outputs.last_hidden_state

    reps = weighted_mean_pooling(hidden, attention_mask)   
    embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
    return embeddings

model_name_or_path = "/media/mldadmin/home/s124mdg39_04/models/VisRAG-Ret"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
#model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
model = model.half().cuda()
model.eval()

script_dir = os.path.dirname(os.path.realpath(__file__))
queries = ["What does a dog look like?"]
passages = [
    Image.open(os.path.join(script_dir, '/media/mldadmin/home/s124mdg39_04/VisRAG/images_test/cat.png')).convert('RGB'),
    Image.open(os.path.join(script_dir, '/media/mldadmin/home/s124mdg39_04/VisRAG/images_test/dog.png')).convert('RGB'),
]

INSTRUCTION = "Represent this query for retrieving relevant documents: "
queries = [INSTRUCTION + query for query in queries]

embeddings_query = encode(queries)
embeddings_doc = encode(passages)

scores = (embeddings_query @ embeddings_doc.T)
print(scores.tolist())