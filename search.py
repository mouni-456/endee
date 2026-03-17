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

print("="*55)
print("🤖 Welcome to Mounika's AI Semantic Search Engine!")
print("   Powered by Endee Vector Database")
print("="*55)

embeddings = model.encode(documents)
print("✅ System Ready! Start Searching...\n")

def search(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    print(f"\n🔍 Results for: '{query}'")
    print("-" * 45)
    for i, idx in enumerate(top_indices):
        score = similarities[idx] * 100
        if score >= 50:
            emoji = "🟢"
        elif score >= 30:
            emoji = "🟡"
        else:
            emoji = "🔴"
        print(f"{emoji} {i+1}. {documents[idx]}")
        print(f"   📊 Relevance: {score:.1f}%")

print("💡 Categories you can search:")
print("   👨‍💻 Jobs: Python job, Java developer, AI engineer")
print("   🧠 Skills: machine learning, NLP, deep learning")
print("   🗄️ Technology: vector database, embeddings, RAG")

while True:
    print()
    query = input("❓ Type your question (or 'quit' to exit): ")
    if query.lower() == 'quit':
        print("\n👋 Thank you for using Mounika's Search Engine!")
        break
    search(query)