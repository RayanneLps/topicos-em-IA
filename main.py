from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import os

# Caminho do banco vetorial
CAMINHO_DB = "chroma"

# Embeddings (mesmo modelo usado na criação do banco)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# LLM local
llm = OllamaLLM(model="llama3")

# Prompt template para RAG
PROMPT_TEMPLATE = """
Responda à pergunta usando APENAS o contexto abaixo:

Contexto:
{contexto}

Pergunta: {pergunta}

Se não encontrar a resposta, diga: "Informação não encontrada nos documentos."
"""

def carregar_db():
    """Carrega o banco Chroma existente"""
    if not os.path.exists(CAMINHO_DB):
        print(f"❌ Banco de dados '{CAMINHO_DB}' não encontrado!")
        print("💡 Execute primeiro 'create_db.py' para criar o banco.")
        return None
    db = Chroma(
        persist_directory=CAMINHO_DB,
        embedding_function=embeddings,
        collection_name="db1"
    )
    print("✅ Banco de dados carregado com sucesso!")
    return db

def responder(pergunta, db):
    """Busca documentos e gera resposta usando RAG + LLM local"""
    # Busca documentos relevantes
    docs = db.similarity_search(pergunta, k=15)
    if not docs:
        return "❌ Nenhum documento relevante encontrado."

    contexto = "\n\n".join([d.page_content for d in docs])
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)

    # Gera resposta com OllamaLLM
    resposta = llm.invoke(prompt)
    return resposta

def main():
    print("🤖 Chatbot RAG com Ollama iniciado!")
    print("📌 Digite 'sair', 'exit' ou 'quit' para encerrar.")
    print("=" * 50)

    db = carregar_db()
    if not db:
        return

    while True:
        pergunta = input("\n💬 Sua pergunta: ").strip()
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("👋 Até logo!")
            break
        if not pergunta:
            print("❌ Por favor, digite uma pergunta.")
            continue

        print("🔍 Buscando documentos relevantes e gerando resposta...")
        resposta = responder(pergunta, db)
        print(f"\n🤖 Resposta:\n{resposta}\n{'-'*50}")

if __name__ == "__main__":
    main()
