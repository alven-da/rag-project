rag-project/
├── data/                # Your raw JSON files
│   ├── products.json
│   └── warranty.json
├── storage/             # Where FAISS will persist the vector index
├── src/
│   ├── __init__.py
│   ├── loader.py        # JSON parsing & synthesis logic
│   ├── vector_store.py  # FAISS & Embedding logic
│   └── main.py          # Entry point / CLI
├── .env                 # Environment variables (API keys if needed)
├── .gitignore
└── requirements.txt     # Dependency list