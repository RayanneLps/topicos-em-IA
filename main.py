'''onde vai acontecer todo o babado'''
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from openai import OpenAI

client = OpenAI(
    api_key = "api-key" ,
)
load_dotenv()

caminho_db = "db"

prompt_template = """
responda a pergunta do usuario:
{pergunta}

com base nas informações a seguir:

{base_de_conhecimento}
 
 
 se você não encontrar a resposta para a pergunta do usuário nessas informações,  
responda não encontrei a informação requirida """

pergunta = input("Qual a sua pergunta? ")

#carregar db
funcao_embeddings = OpenAIEmbeddings(api_key="api-key")

db = Chroma(persist_directory=caminho_db, embedding_function=funcao_embeddings)
#comparar pergunta com db

resultados = db.similarity_search_with_relevance_scores(pergunta)
print(len(resultados))