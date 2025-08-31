from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente
load_dotenv()

# Configurar Gemini Pro
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Caminho do banco de dados vetorial
caminho_db = "chroma"

# Template do prompt para RAG
prompt_template = """
Responda a pergunta do usuário: {pergunta}

Com base nas informações a seguir:
{base_de_conhecimento}

Se você não encontrar a resposta para a pergunta do usuário nessas informações, 
responda "Não encontrei a informação requerida nos documentos fornecidos."

Responda em português e seja claro e objetivo.
"""

def carregar_db():
    """Carrega o banco de dados vetorial"""
    try:
        funcao_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        
        db = Chroma(persist_directory=caminho_db, embedding_function=funcao_embeddings)
        print("✅ Banco de dados carregado com sucesso!")
        return db
    except Exception as e:
        print(f"❌ Erro ao carregar banco de dados: {e}")
        return None

def buscar_documentos_relevantes(db, pergunta, k=3):
    """Busca documentos relevantes no banco vetorial"""
    try:
        resultados = db.similarity_search_with_relevance_scores(pergunta, k=k)
        print(f"🔍 Encontrados {len(resultados)} documentos")
        
        # Filtrar apenas resultados com relevância > 0.7
        documentos_filtrados = [doc for doc, score in resultados if score > 0.7]
        
        if not documentos_filtrados:
            print("⚠️ Nenhum documento relevante com alta confiança")
            # Se não encontrar com alta relevância, usar os 2 melhores mesmo assim
            documentos_filtrados = [doc for doc, score in resultados[:2]]
        
        # Combinar o conteúdo dos documentos relevantes
        contexto = "\n\n".join([doc.page_content for doc in documentos_filtrados])
        return contexto
        
    except Exception as e:
        print(f"❌ Erro na busca: {e}")
        return ""

def gerar_resposta_gemini(pergunta, contexto):
    """Gera resposta usando Gemini Pro com o contexto dos documentos"""
    try:
        # Configurar o modelo
        model = genai.GenerativeModel('gemini-pro')
        
        # Criar o prompt completo
        prompt_completo = prompt_template.format(
            pergunta=pergunta,
            base_de_conhecimento=contexto
        )
        
        # Gerar resposta
        response = model.generate_content(
            prompt_completo,
            generation_config={
                'temperature': 0.1,
                'max_output_tokens': 500,
                'top_p': 0.8,
                'top_k': 10
            }
        )
        
        return response.text
        
    except Exception as e:
        print(f"❌ Erro ao gerar resposta: {e}")
        return "Desculpe, ocorreu um erro ao processar sua pergunta."

def main():
    """Função principal do chatbot"""
    print("🤖 Chatbot RAG com Gemini Pro iniciado!")
    print("Digite 'sair' para encerrar.")
    print("=" * 50)
    
    # Carregar banco de dados
    db = carregar_db()
    if not db:
        print("❌ Não foi possível carregar o banco de dados.")
        print("Certifique-se de que você executou o create_db.py primeiro.")
        return
    
    # Loop principal do chat
    while True:
        pergunta = input("\n💬 Sua pergunta: ").strip()
        
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            print("👋 Até logo!")
            break
        
        if not pergunta:
            print("Por favor, faça uma pergunta.")
            continue
        
        print("🔍 Buscando informações relevantes...")
        
        # Buscar documentos relevantes
        contexto = buscar_documentos_relevantes(db, pergunta)
        
        if not contexto:
            print("❌ Não encontrei informações relevantes nos documentos.")
            continue
        
        print("🤔 Gerando resposta com Gemini Pro...")
        
        # Gerar resposta
        resposta = gerar_resposta_gemini(pergunta, contexto)
        
        print(f"\n🤖 Resposta:\n{resposta}")
        print("-" * 50)

if __name__ == "__main__":
    # Verificações iniciais
    if not os.path.exists('.env'):
        print("❌ Arquivo .env não encontrado!")
        print("Crie um arquivo .env com sua GOOGLE_API_KEY")
    
    elif not os.getenv('GOOGLE_API_KEY'):
        print("❌ GOOGLE_API_KEY não encontrada no arquivo .env")
        print("1. Acesse: https://makersuite.google.com/app/apikey")
        print("2. Crie uma API key")
        print("3. Adicione no .env: GOOGLE_API_KEY=sua_chave")
    
    elif not os.path.exists(caminho_db):
        print(f"❌ Banco de dados '{caminho_db}' não encontrado!")
        print("Execute primeiro o create_db.py para criar o banco.")
    
    else:
        main()