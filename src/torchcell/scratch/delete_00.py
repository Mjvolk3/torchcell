import json

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Read the data
with open("data/bary_data.json", "r") as f:
    data = json.load(f)
sequences = [entry["sequence"] for entry in data.values()]

# Nucleotide Transformer
tokenizer = AutoTokenizer.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
)
model = AutoModelForMaskedLM.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
)

# Initialize a list to hold mean sequence embeddings
mean_sequence_embeddings = []

# Save every N sequences
N = 100

# Loop through sequences and process them one by one
for idx, sequence in enumerate(tqdm(sequences)):
    tokens_ids = tokenizer.encode_plus(
        sequence, return_tensors="pt", truncation=True, padding=True
    )["input_ids"]
    attention_mask = tokens_ids != tokenizer.pad_token_id
    torch_outs = model(
        tokens_ids, attention_mask=attention_mask, output_hidden_states=True
    )

    # Compute the mean embedding for this sequence
    embedding = torch_outs["hidden_states"][-1].detach()
    mean_embedding = torch.sum(
        attention_mask.unsqueeze(-1) * embedding, axis=-2
    ) / torch.sum(attention_mask, axis=-1)
    mean_sequence_embeddings.append(mean_embedding.squeeze().tolist())

    # Save every N iterations
    if (idx + 1) % N == 0:
        for i, key in enumerate(list(data.keys())[: idx + 1]):
            data[key]["embedding"] = mean_sequence_embeddings[i]
        with open(f"data/bary_embed_{idx + 1}.json", "w") as f:
            json.dump(data, f)

# Dump the final data dictionary to a new JSON file
with open("data/bary_embed_all.json", "w") as f:
    json.dump(data, f)
