from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Python developer with machine learning experience needed",
    "Data scientist role requiring NLP and deep learning skills",
    "AI engineer position working with vector databases",
    "Software engineer building recommendation systems using AI",
    "Backend developer with experience in REST APIs and Python",
    "Machine learning engineer working on computer vision",
    "Java developer with spring boot and microservices experience",
    "Web developer with HTML CSS and JavaScript skills",
    "Endee vector database is used for high performance AI search",
    "Semantic search helps find meaning beyond exact keywords",
    "Vector embeddings represent text as mathematical numbers",
    "RAG systems combine vector search with language models",
    "Python is best language for machine learning and AI",
    "NLP helps computers understand human language",
    "Deep learning uses neural networks to solve complex problems"
]

embeddings = model.encode(documents)

# Chat History Memory
chat_history = []

def search(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    print(f"\nResults for: '{query}'")
    print("-" * 45)
    for i, idx in enumerate(top_indices):
        score = similarities[idx] * 100
        if score >= 50:
            label = "[HIGH MATCH]"
        elif score >= 30:
            label = "[MEDIUM MATCH]"
        else:
            label = "[LOW MATCH]"
        print(f"{i+1}. {documents[idx]}")
        print(f"   Relevance: {score:.1f}% {label}")
        results.append(documents[idx])
    
    # Save to memory
    chat_history.append({
        "question": query,
        "top_result": results[0] if results else ""
    })
    return results

def show_history():
    if not chat_history:
        print("\nNo history yet!")
        return
    print("\n=== Your Chat History ===")
    for i, item in enumerate(chat_history):
        print(f"{i+1}. You asked: {item['question']}")
        print(f"   Best result: {item['top_result']}")
    print("="*45)

print("="*55)
print("Welcome to Mounika's AI Semantic Search Engine!")
print("Powered by Endee Vector Database")
print("With Chat History and Memory!")
print("="*55)
print("\nCommands:")
print("   Type any question to search")
print("   Type 'history' to see past searches")
print("   Type 'quit' to exit")

while True:
    print()
    query = input("Your question: ")
    if query.lower() == 'quit':
        print("Thank you for using Mounika's Search Engine!")
        break
    elif query.lower() == 'history':
        show_history()
    else:
        search(query)