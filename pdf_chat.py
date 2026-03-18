from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Abstraction means hiding method implementation details and exposing method signature",
    "Abstraction can be achieved by using interface or abstract class",
    "Steps to achieve abstraction: create interface, abstract method, implementation class",
    "Helper method name must be similar to interface name and begin with get",
    "Inbuilt library consists of inbuilt classes and inbuilt interfaces",
    "Java packages include java.lang, java.util, java.io",
    "System.out.println uses System class, out variable and println method",
    "String is a final class in java.lang package which cannot be inherited",
    "String immutable property means string data cannot be changed",
    "ArrayList maintains insertion order and allows duplicate objects",
    "LinkedList memory blocks are scattered randomly in double linked fashion",
    "ArrayList is faster when elements are added or removed at the end",
    "LinkedList is faster when elements are added or removed in between",
    "Vector objects are thread safe because methods are synchronized",
    "Queue stores objects in FIFO fashion using LinkedList",
    "Set does not allow duplicate objects and allows only one null value",
    "HashSet is unordered, LinkedHashSet maintains insertion order, TreeSet sorting order",
    "Wrapper classes represent primitive datatype as object",
    "Boxing is converting primitive data into object, unboxing is converting back",
    "Arrays are objects in Java with fixed size and homogeneous types",
    "hashcode method returns hashcode number for object address",
    "equals method compares objects based on object address",
    "toString method returns complete information of current object",
]

embeddings = model.encode(documents)
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
    print("=" * 45)

print("=" * 55)
print("Welcome to Mounika's AI Java Notes Search!")
print("Powered by Endee Vector Database")
print("=" * 55)
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