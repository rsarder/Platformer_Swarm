import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def embed_json(filename="mechanics.json"):
    # Load the input JSON file with your mechanics (e.g., 'mechanics.json')
    with open('mechanics.json', 'r') as infile:
        mechanics = json.load(infile)

    # Initialize the BERT-based model; here we use a lightweight model for speed
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute an embedding for each mechanic and add it as a new field
    for mechanic in mechanics:
        # Combine fields for a richer representation
        text = f"{mechanic['Name']}. {mechanic['Description']}"
        embedding = model.encode(text).tolist()  # Convert numpy array to list for JSON serialization
        mechanic['embedding'] = embedding

    # Save the new JSON with embeddings to a file
    with open('mechanics_with_embeddings.json', 'w') as outfile:
        json.dump(mechanics, outfile, indent=2)

    print("Embeddings created and saved to 'mechanics_with_embeddings.json'.")
    
    

def search_mechanics_dynamic(query, initial_top_k=10, threshold=1.5):
    # Load the JSON file with embeddings
    with open('mechanics_with_embeddings.json', 'r') as infile:
        mechanics = json.load(infile)

    # Prepare a NumPy array of embeddings from the mechanics
    embeddings = np.array([m['embedding'] for m in mechanics]).astype('float32')

    # Create a FAISS index (using L2 distance here)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Add our vectors to the index

    # Initialize the same SentenceTransformer model for queries
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute the query embedding using the same model
    query_embedding = model.encode(query).astype('float32')
    query_embedding = np.expand_dims(query_embedding, axis=0)
    
    # Retrieve a larger set of candidates initially
    distances, indices = index.search(query_embedding, initial_top_k)
    
    # Convert distances to similarities if needed (e.g., if using cosine similarity, similarity = 1 - distance)
    # Here, we assume a threshold where lower distance means more similar.
    relevant_results = []
    for dist, idx in zip(distances[0], indices[0]):
        # Apply a threshold: adjust condition based on how your distances are measured
        if dist < threshold:
            relevant_results.append(mechanics[idx])
    
    return relevant_results

# embed_json()
# Usage example:
query = input("Enter your query: ")
results = search_mechanics_dynamic(query, initial_top_k=15, threshold=1.5)
if results:
    for idx, result in enumerate(results, 1):
        print(f"Result {idx}:")
        print("Name:", result["Name"])
        print("Description:", result["Description"])
        print("Implementation Details:", result["Implementation Details"])
        print("Phaser Pseudocode:", result["Pseudocode (Phaser.js)"])
        print("-----")
else:
    print("No relevant mechanics found for your query.")
