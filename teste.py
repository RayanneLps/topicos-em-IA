from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(
    persist_directory="chroma",
    embedding_function=embeddings,
    collection_name="db1"
)

# Faça uma pergunta real sobre seus PDFs
pergunta = "qual o valor da bolsa pibic"
docs = db.similarity_search(pergunta, k=5)

print(f"📊 Encontrados: {len(docs)} documentos\n")
for i, doc in enumerate(docs, 1):
    print(f"--- Documento {i} ---")
    print(doc.page_content[:300])
    print("\n")