from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

''' lê o arquivo .env e carrega a chave da API da OPENAI. 
    reads the .env file and loads the OPENAI API key.
'''
load_dotenv() 

BASE_FOLDER = "base"

def create_db():
    documents =  load_documents()
    chucks = split_documents(documents)
    vectorize_chuncks(chucks)
    
def load_documents():
    ''' ler meus documentos PDFs da pasta base e retornar uma lista de documentos.
        read my PDF documents from the base folder and return a list of documents.
    '''
    looader_docs = PyPDFDirectoryLoader(BASE_FOLDER, glob="*.pdf")
    documents = looader_docs.load()
    return documents

def split_documents(documents):
    ''' dividir os documentos em pedaços menores(chunks).
        split documents into smaller chunks.
    '''
    separator_doc = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # 1 chunk(pedaço) = 1000 characters.
        chunk_overlap=500,      # chunks vão se sobrepor começando em 500 caracteres antes. The chunks will overlap starting 500 characters before.
        length_function=len,
        add_start_index=True
    )
    chunks = separator_doc.split_documents(documents)
    return chunks
    

def vectorize_chuncks(chucks):
    db = Chroma.from_documents(chucks, OpenAIEmbeddings(), collection_name="db")
    print("Database created successfully!")
    
    
create_db()