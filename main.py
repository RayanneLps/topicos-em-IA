'''onde vai acontecer todo o babado'''
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from openai import OpenAI

client = OpenAI(
    api_key = "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" ,
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
funcao_embeddings = OpenAIEmbeddings(api_key="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")

db = Chroma(persist_directory=caminho_db, embedding_function=funcao_embeddings)
#comparar pergunta com db

resultados = db.similarity_search_with_relevance_scores(pergunta)
print(len(resultados))