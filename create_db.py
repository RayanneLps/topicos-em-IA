from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from preprocessar_pdfs import carregar_e_preprocessar_pdfs
import os
import shutil
import time

BASE_FOLDER = "base"   # pasta com PDFs
CHROMA_PATH = "chroma" # pasta do banco vetorial
BATCH_SIZE = 20  # Aumentado para melhor performance após pré-processamento

def criar_db():
    print("Iniciando criacao do banco de dados com OllamaEmbeddings...")

    # Apagar banco antigo se existir
    if os.path.exists(CHROMA_PATH):
        print(f"Removendo banco antigo em '{CHROMA_PATH}'...")
        shutil.rmtree(CHROMA_PATH)

    # Carregar e pré-processar documentos
    if not os.path.exists(BASE_FOLDER):
        print(f"ERRO: Pasta '{BASE_FOLDER}' nao encontrada!")
        return

    # Usar pré-processamento otimizado
    documentos = carregar_e_preprocessar_pdfs(BASE_FOLDER)
    if not documentos:
        print("ERRO: Nenhum documento carregado!")
        return

    print(f"OK: {len(documentos)} documentos carregados e preprocessados")
    
    # Dividir em chunks otimizados (chunks maiores após pré-processamento)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,      # Aumentado para melhor contexto
        chunk_overlap=300,    # Overlap otimizado
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]  # Priorizar quebras naturais
    )
    chunks = splitter.split_documents(documentos)
    print(f"OK: {len(chunks)} chunks criados")

    # Criar embeddings Ollama
    print("Configurando embeddings Ollama...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Criar banco Chroma vazio primeiro
    print("Criando banco de dados Chroma...")
    db = Chroma(
        embedding_function=embeddings,
        collection_name="db1",
        persist_directory=CHROMA_PATH
    )

    # Processar chunks em lotes para evitar timeout
    print(f"Processando {len(chunks)} chunks em lotes de {BATCH_SIZE}...")
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"Processando lote {batch_num}/{total_batches} (chunks {i+1}-{min(i+BATCH_SIZE, total_chunks)})...")
        
        try:
            # Adicionar batch ao banco
            db.add_documents(batch)
            print(f"OK: Lote {batch_num} processado com sucesso")
        except KeyboardInterrupt:
            print("\nProcesso interrompido pelo usuario")
            raise
        except Exception as e:
            print(f"ERRO ao processar lote {batch_num}: {e}")
            print("Tentando novamente...")
            time.sleep(2)  # Esperar um pouco antes de tentar novamente
            try:
                db.add_documents(batch)
                print(f"OK: Lote {batch_num} processado com sucesso (na segunda tentativa)")
            except Exception as e2:
                print(f"ERRO persistente no lote {batch_num}: {e2}")
                raise

    print(f"\nOK: Banco de dados criado com sucesso e salvo em: {CHROMA_PATH}")
    print(f"Total de {total_chunks} chunks indexados")

if __name__ == "__main__":
    criar_db()