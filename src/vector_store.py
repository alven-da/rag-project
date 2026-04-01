import os

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

class VectorStoreManager:
    def __init__(self, index_path: str = "storage/faiss_index"):
        self.index_path = index_path
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    def create_and_save(self, documents: List[Document]):
        """Converts text to vectors and persists them to disk."""
        print(f"Creating vector index for {len(documents)} documents...")
        
        # This step calls Ollama to generate embeddings for each document
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Persist to the 'storage/' directory so we don't re-embed every time
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        vectorstore.save_local(self.index_path)
        print(f"Index saved to {self.index_path}")
        return vectorstore
    
    def load_index(self):
        """Loads the existing index from disk."""
        if os.path.exists(self.index_path):
            # In this version, we don't need the allow_dangerous_deserialization flag
            return FAISS.load_local(
                self.index_path, 
                self.embeddings
            )
        return None

    def search_with_filter(self, query: str, category_name: str, k: int = 2):
        """Run a similarity search filtered by category metadata."""
        vs = self.load_index()
        
        # FAISS uses a simple dictionary filter for metadata
        # This says: "Only look at docs where metadata['category'] == category_name"
        search_kwargs = {
            "filter": {"category": category_name},
            "k": k
        }
    
        return vs.similarity_search(query, **search_kwargs)
    
if __name__ == "__main__":
    from loader import JSONLoader
    
    loader = JSONLoader('data/products.json', 'data/warranty.json')
    
    manager = VectorStoreManager()

    vs = manager.load_index()
    if not vs:
        docs = loader.create_unified_doc()
        vs = manager.create_and_save(docs)

    print("\n--- Vector Store CLI Tool ---")
    print("Type 'exit' to quit. Ask about product specs or warranty terms.")

    while True:
        user_prompt = input("\n[Search Query]: ")

        if user_prompt.lower() in ['exit', 'quit', 'q']:
            break

        results = vs.similarity_search(user_prompt, k=1)

        if results:
            match = results[0]
            print(f"\n--- Top Semantic Match (SKU: {match.metadata['sku']}) ---")
            print(f"Synthesized Content:\n{match.page_content}")
            print(f"Match Confidence Score: (Internal FAISS distance)")
        else:
            print("No matches found.")
