from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

'''
lê o arquivo .env e carrega a chave da API do Google Gemini.     
reads the .env file and loads the Google Gemini API key.
'''
load_dotenv()

BASE_FOLDER = "base"  # Sua pasta com PDFs
CHROMA_PATH = "chroma"  # Onde será salvo o banco

def create_db():
    print("🚀 Iniciando criação do banco com Gemini Pro...")
    documents = load_documents()
    
    if not documents:
        print("❌ Nenhum documento carregado!")
        return
        
    chunks = split_documents(documents)
    
    if not chunks:
        print("❌ Nenhum chunk criado!")
        return
        
    vectorize_chunks(chunks)

def load_documents():
    '''
    ler meus documentos PDFs da pasta base e retornar uma lista de documentos.
    read my PDF documents from the base folder and return a list of documents.
    '''
    print(f"📁 Carregando documentos de: {BASE_FOLDER}")
    
    if not os.path.exists(BASE_FOLDER):
        print(f"❌ Pasta '{BASE_FOLDER}' não encontrada!")
        return []
    
    try:
        loader_docs = PyPDFDirectoryLoader(BASE_FOLDER, glob="**/*.pdf")
        documents = loader_docs.load()
        print(f"✅ {len(documents)} documentos carregados")
        
        # Mostrar preview
        for i, doc in enumerate(documents[:3]):
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"📖 Doc {i+1}: {content_preview}...")
            
        return documents
        
    except Exception as e:
        print(f"❌ Erro ao carregar documentos: {e}")
        return []

def split_documents(documents):
    '''
    dividir os documentos em pedaços menores(chunks).
    split documents into smaller chunks.
    '''
    print("✂️ Dividindo documentos em chunks...")
    
    try:
        separator_doc = RecursiveCharacterTextSplitter(
            chunk_size=1500,        # 1 chunk = 1000 characters
            chunk_overlap=150,      # chunks se sobrepõem em 500 caracteres
            length_function=len,
            add_start_index=True
        )
        
        chunks = separator_doc.split_documents(documents)
        print(f"✅ {len(chunks)} chunks criados")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Erro ao criar chunks: {e}")
        return []

def vectorize_chunks(chunks):
    '''
    Criar embeddings usando Gemini Pro e salvar no banco vetorial Chroma.
    '''
    print("🧠 Criando embeddings com Gemini Pro...")
    
    try:
        # Verificar se a API key do Google está configurada
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY não encontrada no arquivo .env!")
            print("Adicione sua chave do Google AI Studio no arquivo .env:")
            print("GOOGLE_API_KEY=sua_chave_aqui")
            return
        
        # Criar embeddings usando Gemini
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Criar banco vetorial
        db = Chroma.from_documents(
            chunks, 
            embeddings, 
            collection_name="db1",
            persist_directory=CHROMA_PATH
        )
        
        print("✅ Banco de dados criado com sucesso usando Gemini Pro!")
        print(f"📍 Salvo em: {CHROMA_PATH}")
        print(f"📊 Total de chunks: {len(chunks)}")
        
    except Exception as e:
        print(f"❌ Erro ao criar embeddings: {e}")
        print("Verifique:")
        print("- Se a GOOGLE_API_KEY está correta")
        print("- Se você tem acesso à API do Google AI")

def main():
    print("=" * 50)
    print("🤖 CRIADOR DE BANCO VETORIAL - GEMINI PRO")
    print("=" * 50)
    
    # Verificações
    if not os.path.exists('.env'):
        print("❌ Arquivo .env não encontrado!")
        return
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ GOOGLE_API_KEY não encontrada!")
        print("1. Vá para: https://makersuite.google.com/app/apikey")
        print("2. Crie uma API key")
        print("3. Adicione no .env: GOOGLE_API_KEY=sua_chave")
        return
    
    create_db()

if __name__ == "__main__":
    main()