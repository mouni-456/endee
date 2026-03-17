from sentence_transformers import SentenceTransformer
import numpy as np

# Load AI model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample data - documents we want to search
documents = [
    "Python is a popular programming language for AI",
    "Machine learning helps computers learn from data",
    "Vector databases store data as numbers called embeddings",
    "Endee is a high performance vector database",
    "Semantic search finds meaning not just keywords",
    "Natural language processing helps computers understand text"
]

# Convert documents to embeddings
print("Loading documents into Endee...")
embeddings = model.encode(documents)
print(f"Loaded {len(documents)} documents successfully!")

# Search function
def search(query, top_k=3):
    query_embedding = model.encode([query])
    
    # Calculate similarity
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\nSearch Results for: '{query}'")
    print("-" * 40)
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {documents[idx]}")
        print(f"   Score: {similarities[idx]:.3f}")

# Test the search
search("what is vector database")
search("how does AI learn")