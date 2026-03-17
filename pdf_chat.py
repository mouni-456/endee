from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

def read_pdf(pdf_path):
    text_chunks = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                sentences = text.split('.')
                for sentence in sentences:
                    if len(sentence.strip()) > 20:
                        text_chunks.append(sentence.strip())
    return text_chunks

def chat_with_pdf(pdf_path):
    print("="*55)
    print("AI-Based Document Query System")
    print("Powered by Endee Vector Database")
    print("="*55)
    
    print(f"\nReading PDF: {pdf_path}")
    chunks = read_pdf(pdf_path)
    
    if not chunks:
        print("Could not read PDF!")
        return
    
    print(f"Found {len(chunks)} text sections!")
    print("Building search index...")
    embeddings = model.encode(chunks)
    print("Ready! Ask me anything about your PDF!\n")
    
    while True:
        query = input("Your question (or quit to exit): ")
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        query_embedding = model.encode([query])
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print(f"\nAnswer from PDF:")
        print("-" * 45)
        for i, idx in enumerate(top_indices):
            score = similarities[idx] * 100
            if score >= 30:
                print(f"{i+1}. {chunks[idx]}")
                print(f"   Relevance: {score:.1f}%")
        print()

pdf_file = input("Enter PDF file path (or press Enter for demo): ")
if pdf_file == "":
    print("\nDemo mode - No PDF uploaded")
    print("In real use, provide a PDF path like: C:/Users/yourname/document.pdf")
else:
    if os.path.exists(pdf_file):
        chat_with_pdf(pdf_file)
    else:
        print("PDF file not found!")