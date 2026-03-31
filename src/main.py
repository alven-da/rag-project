import sys
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from vector_store import VectorStoreManager

def start_chat():
    # 1. Initialize the Local Brain (Llama 3)
    # This calls your local Ollama server
    llm = Ollama(model="llama3")

    # 2. Load the Knowledge Base (FAISS)
    vs_manager = VectorStoreManager()
    vectorstore = vs_manager.load_index()

    if not vectorstore:
        print("Error: Vector index not found. Please run 'python src/vector_store.py' first.")
        return

    template = """
    You are a professional Technical Support Assistant. 
    Use the following pieces of retrieved context to answer the user's question. 
    If the answer isn't in the context, politely say you don't have that information.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # 4. Create the Retrieval Chain
    # 'k=2' means it will grab the top 2 most relevant merged JSON records
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True # This allows us to see which SKU was used
    )

    print("\n--- RAG Support System Ready (Llama 3) ---")
    print("Ask me about product specs or warranty coverage. (Type 'exit' to quit)")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break

        # Execute the RAG logic
        result = qa_chain.invoke({"query": user_input})
        
        print(f"\nAI: {result['result']}")
        
        # Senior Tip: Observability. Show which SKU the AI looked at.
        sources = [doc.metadata.get('sku') for doc in result['source_documents']]
        print(f"\n[Source Context: {', '.join(sources)}]")

if __name__ == "__main__":
    start_chat()