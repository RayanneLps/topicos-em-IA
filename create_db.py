from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os
import shutil

BASE_FOLDER = "base"   # pasta com PDFs
CHROMA_PATH = "chroma" # pasta do banco vetorial

def criar_db():
    print("🚀 Iniciando criação do banco de dados com OllamaEmbeddings...")

    # Apagar banco antigo se existir
    if os.path.exists(CHROMA_PATH):
        print(f"🧹 Removendo banco antigo em '{CHROMA_PATH}'...")
        shutil.rmtree(CHROMA_PATH)

    # Carregar documentos
    if not os.path.exists(BASE_FOLDER):
        print(f"❌ Pasta '{BASE_FOLDER}' não encontrada!")
        return

    loader = PyPDFDirectoryLoader(BASE_FOLDER, glob="**/*.pdf")
    documentos = loader.load()
    if not documentos:
        print("❌ Nenhum documento carregado!")
        return

    print(f"✅ {len(documentos)} documentos carregados")
    
    # Dividir em chunks
    splitter = RecursiveCharacterTextSplitter(
        # chunk_size=3000, #versao do marcelo
        # chunk_overlap=600,
        # length_function=len,
        # add_start_index=True,
        #############################################
        chunk_size=800,      
        chunk_overlap=200,   
        length_function=len,
        add_start_index=True, #versao funcional 1
        separators=["\n\n", "\n", ". ", " ", ""] 
    )
    chunks = splitter.split_documents(documentos)
    print(f"✅ {len(chunks)} chunks criados")

    # Mostrar os primeiros 3 chunks com preview
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.page_content[:300].replace('\n', ' ')
        print(f"Chunk {i+1} preview: {preview}...\n")

    # Criar embeddings Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Criar banco Chroma
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="db1",
        persist_directory=CHROMA_PATH
    )

    print("✅ Banco de dados criado com sucesso e salvo em:", CHROMA_PATH)

if __name__ == "__main__":
    criar_db()