import ollama

try:
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': 'Is the environment ready?'}
    ])
    print("Ollama Connection: SUCCESS")
    print(f"Llama3 says: {response['message']['content']}")
except Exception as e:
    print(f"Ollama Connection: FAILED. Error: {e}")