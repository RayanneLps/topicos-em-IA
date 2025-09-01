from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time
import hashlib

# Carregar variáveis de ambiente
load_dotenv()

# Configurar Gemini Pro
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Caminho do banco de dados vetorial
caminho_db = "chroma"

# Template do prompt para RAG (otimizado)
prompt_template = """
Sua tarefa é responder a uma pergunta usando APENAS as informações contidas no 'Contexto' abaixo.
Sua resposta deve ser objetiva e em português.

Contexto:
{base_de_conhecimento}

Pergunta: {pergunta}

Se a resposta para a pergunta não puder ser encontrada no contexto, responda "Informação não encontrada nos documentos."
"""

# Cache melhorado com hash
respostas_cache = {}

# Controle de rate limiting
ultima_requisicao = 0
INTERVALO_MIN_REQUISICOES = 2  # segundos entre requisições

def criar_hash_pergunta(pergunta):
    """Cria hash da pergunta para cache mais eficiente"""
    return hashlib.md5(pergunta.lower().strip().encode()).hexdigest()

def aguardar_rate_limit():
    """Controla o rate limiting"""
    global ultima_requisicao
    tempo_atual = time.time()
    tempo_decorrido = tempo_atual - ultima_requisicao
    
    if tempo_decorrido < INTERVALO_MIN_REQUISICOES:
        tempo_espera = INTERVALO_MIN_REQUISICOES - tempo_decorrido
        print(f"⏱️ Aguardando {tempo_espera:.1f}s para evitar rate limit...")
        time.sleep(tempo_espera)
    
    ultima_requisicao = time.time()

def carregar_db():
    """Carrega o banco de dados vetorial"""
    try:
        funcao_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        
        db = Chroma(persist_directory=caminho_db, embedding_function=funcao_embeddings, collection_name="db1")
        print("✅ Banco de dados carregado com sucesso!")
        return db
    except Exception as e:
        print(f"❌ Erro ao carregar banco de dados: {e}")
        return None

def buscar_documentos_relevantes(db, pergunta, k=5):
    """Busca documentos relevantes (aumentado para k=5)"""
    try:
        resultados = db.similarity_search_with_relevance_scores(pergunta, k=k)
        print(f"🔍 Encontrados {len(resultados)} documentos")
        
        # Filtrar apenas resultados com relevância > 0.6 (reduzido o threshold)
        documentos_filtrados = [doc for doc, score in resultados if score > 0.6]
        
        if not documentos_filtrados:
            print("⚠️ Nenhum documento relevante encontrado")
            return ""
        
        # Combinar e limitar o tamanho do contexto
        contexto = "\n\n".join([doc.page_content for doc in documentos_filtrados])
        return contexto
        
    except Exception as e:
        print(f"❌ Erro na busca: {e}")
        return ""

def gerar_resposta_gemini(pergunta, contexto):
    """Gera resposta usando Gemini com otimizações"""
    try:
        # Verificar cache com hash
        hash_pergunta = criar_hash_pergunta(pergunta)
        if hash_pergunta in respostas_cache:
            print("💾 Resposta encontrada no cache.")
            return respostas_cache[hash_pergunta]

        # Rate limiting
        aguardar_rate_limit()

        # MUDANÇA PRINCIPAL: Usar gemini-1.5-flash (mais leve e com maior cota gratuita)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt_completo = prompt_template.format(
            pergunta=pergunta,
            base_de_conhecimento=contexto
        )

        response = model.generate_content(
            prompt_completo,
            generation_config={
                # Reduzido para 256 tokens para maior probabilidade de funcionar com cota gratuita
                'max_output_tokens': 256,
                # Ajustado para maior determinismo
                'temperature': 0.1,
                'top_p': 0.8,
                'top_k': 10
            }
        )
        
        # Salvar no cache
        respostas_cache[hash_pergunta] = response.text
        
        return response.text
        
    except Exception as e:
        erro_str = str(e)
        if "429" in erro_str or "quota" in erro_str.lower():
            print("❌ Cota da API excedida!")
            print("💡 Soluções:")
            print("1. Aguarde algumas horas/dia para resetar a cota gratuita")
            print("2. Use um modelo mais leve (gemini-1.5-flash)")
            print("3. Considere fazer upgrade para um plano pago")
            return "Cota da API excedida. Tente novamente mais tarde."
        else:
            print(f"❌ Erro ao gerar resposta: {e}")
            return "Erro ao processar sua pergunta."

def verificar_cota_disponivel():
    """Testa se a API está funcionando com uma requisição simples"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            "Diga apenas 'ok'", 
            generation_config={'max_output_tokens': 5}
        )
        return True
    except Exception as e:
        if "429" in str(e):
            return False
        return True

def main():
    """Função principal do chatbot"""
    print("🤖 Chatbot RAG com Gemini Flash iniciado!")
    print("Digite 'sair' para encerrar.")
    print("=" * 50)
    
    # Verificar cota antes de começar
    if not verificar_cota_disponivel():
        print("❌ Cota da API excedida no momento.")
        print("⏰ Aguarde algumas horas e tente novamente.")
        return
    
    db = carregar_db()
    if not db:
        print("❌ Não foi possível carregar o banco de dados.")
        return
    
    contador_perguntas = 0
    
    while True:
        pergunta = input("\n💬 Sua pergunta: ").strip()
        
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            print("👋 Até logo!")
            break
        
        if not pergunta:
            print("Por favor, faça uma pergunta.")
            continue
        
        contador_perguntas += 1
        
        # Avisar sobre limite após algumas perguntas
        if contador_perguntas > 10:
            print("⚠️ Você já fez muitas perguntas. Se der erro 429, aguarde um tempo.")
        
        # Verificar cache primeiro
        hash_pergunta = criar_hash_pergunta(pergunta)
        if hash_pergunta in respostas_cache:
            print(f"\n🤖 Resposta (do cache):\n{respostas_cache[hash_pergunta]}")
            print("-" * 50)
            continue

        print("🔍 Buscando informações relevantes...")
        
        contexto = buscar_documentos_relevantes(db, pergunta)
        
        if not contexto:
            print("❌ Não encontrei informações relevantes nos documentos.")
            continue
        
        print("🤔 Gerando resposta com Gemini Flash...")
        
        resposta = gerar_resposta_gemini(pergunta, contexto)
        
        print(f"\n🤖 Resposta:\n{resposta}")
        print("-" * 50)

if __name__ == "__main__":
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
