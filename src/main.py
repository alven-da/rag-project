import sys

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from vector_store import VectorStoreManager

def start_chat():
    llm = Ollama(model="llama3")

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

    print("\n--- RAG Support System (Filtered Search) ---")
    print("Categories: Power Tools, Home Security, Electronics, Appliances, Home Improvement")
    print("Type 'exit' to quit.")

    while True:
        cat_input = input("\n[Filter Category] e.g. Power Tools | Home Security | Electronics | Appliances | Home Improvement (or press Enter for All): ").strip()
        
        user_query = input("[Your Query]: ").strip()
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            break

        # intial value for search_kwargs
        search_kwargs = {"k": 2}

        # if there is an input for category, add the filter to search_kwargs
        # assuming the category names in metadata are exactly as listed in the prompt (case-sensitive)
        if cat_input:
            search_kwargs["filter"] = {"category": cat_input}
            print(f"--- Searching only in {cat_input} ---")

        # Step D: Re-initialize the chain with the specific filter
        # In a production API, you'd do this per request
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs=search_kwargs),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )

        # Step E: Execute
        result = qa_chain.invoke({"query": user_query})
        
        print(f"\nAI: {result['result']}")
        
        # Observability: Confirm which SKUs were pulled
        sources = [doc.metadata.get('sku') for doc in result['source_documents']]
        print(f"[Verified Sources: {', '.join(sources)}]")

if __name__ == "__main__":
    start_chat()