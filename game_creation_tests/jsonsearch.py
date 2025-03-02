import json
import numpy as np
import faiss
import torch
from transformers import BertTokenizer, BertModel

# ---------- Configuration ----------
MECHANICS_FILE = 'mechanics.json'
INDEXED_FILE = 'mechanics_with_embeddings.json'
TOP_K = 1  # number of nearest neighbors to retrieve

# ---------- BERT Embedding Functions ----------
# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed_text(text):
    """
    Convert input text into an embedding using BERT.
    Uses the [CLS] token representation as the sentence embedding.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the [CLS] token embedding (first token) and flatten it
    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    return embedding.tolist()

# ---------- Load or Compute Embeddings for Game Mechanics ----------
def load_mechanics(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_mechanics(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

mechanics = load_mechanics(MECHANICS_FILE)

# Check each mechanic; if no embedding exists, compute it from the key (mechanic name)
for mechanic, data in mechanics.items():
    if 'embedding' not in data:
        data['embedding'] = embed_text(mechanic)

# Save the updated JSON (so next time you won't need to recompute all embeddings)
save_mechanics(INDEXED_FILE, mechanics)

# ---------- Build Faiss Index ----------
# Assume that all embeddings have the same dimension (e.g., 768 for BERT-base)
mechanic_names = list(mechanics.keys())
embeddings_list = [np.array(mechanics[name]['embedding'], dtype=np.float32) for name in mechanic_names]
if not embeddings_list:
    raise ValueError("No embeddings found!")
d = embeddings_list[0].shape[0]

# Create a Faiss index using L2 distance (Euclidean distance)
index = faiss.IndexFlatL2(d)
embeddings_matrix = np.vstack(embeddings_list)  # shape: (num_mechanics, d)
index.add(embeddings_matrix)

# ---------- Query Function ----------
def query_mechanics(query_text, top_k=TOP_K):
    # Compute the embedding for the query text
    query_emb = np.array(embed_text(query_text), dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_emb, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        mechanic = mechanic_names[idx]
        results.append({
            'mechanic': mechanic,
            'distance': float(distances[0][i]),
            'details': mechanics[mechanic]['details']
        })
    return results

# ---------- Example Usage ----------
if __name__ == '__main__':
    query = "flying"
    matches = query_mechanics(query, top_k=TOP_K)
    print(f"Best matching mechanic(s) for query '{query}':")
    for match in matches:
        print(f"Mechanic: {match['mechanic']} (L2 Distance: {match['distance']:.3f})")
        print("Details:", match['details'])
