import json
from langchain.schema import Document
from engine import vectorstore, text_splitter

def ingest_massive_profile(file_path: str):
    print(f"📖 Reading massive profile data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert the entire JSON directly into a highly readable YAML-like string for the LLM
    profile_text = json.dumps(data, indent=2)
    
    # Add a semantic header so the LLM knows what this data represents
    context_header = "THE FOLLOWING IS THE DEFINITIVE BIOGRAPHY, TECHNICAL PROFILE, AND PREFERENCES OF SIR HIMANSHU:\n\n"
    full_content = context_header + profile_text

    doc = Document(
        page_content=full_content,
        metadata={
            "source": "Core Memory Module", 
            "title": "Himanshu Master Database"
        }
    )

    chunks = text_splitter.split_documents([doc])
    vectorstore.add_documents(chunks)
    
    print(f"✅ Success! Injected {len(chunks)} knowledge chunks into ChromaDB.")

if __name__ == "__main__":
    ingest_massive_profile("himanshu_profile.json")