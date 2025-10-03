import gradio as gr
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Configurações
CAMINHO_DB = "chroma"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# Configuração ultra-rápida
llm = OllamaLLM(
    model="llama3.2",
    temperature=0.1,      # Muito focado
    num_ctx=2048,         # Contexto bem menor
    num_predict=256,      # Limita tamanho da resposta
    top_k=10,             # Menos tokens considerados
    top_p=0.9
)

PROMPT_TEMPLATE = """
Responda à pergunta usando APENAS o contexto abaixo:

Contexto: {contexto}

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
    # Busca documentos relevantes (reduzido de 15 para 5)
    docs = db.similarity_search(pergunta, k=5)
    if not docs:
        return "❌ Nenhum documento relevante encontrado."
    
    # Limita o tamanho do contexto (máximo 2000 caracteres por doc)
    contexto_limitado = []
    for doc in docs:
        texto = doc.page_content[:2000]
        contexto_limitado.append(texto)
    
    contexto = "\n\n".join(contexto_limitado)
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)
    
    # Gera resposta com OllamaLLM
    resposta = llm.invoke(prompt)
    return resposta

# Inicializar banco ao carregar
print("🔄 Inicializando sistema...")
db_global = carregar_db()

def responder_pergunta(pergunta, historico):
    """Responde perguntas usando o RAG"""
    if db_global is None:
        resposta = "❌ Banco de dados não inicializado. Execute 'create_db.py' primeiro."
        historico.append((pergunta, resposta))
        return historico, historico
    
    if not pergunta.strip():
        resposta = "⚠️ Por favor, digite uma pergunta!"
        historico.append((pergunta, resposta))
        return historico, historico
    
    try:
        print(f"🔍 Processando pergunta: {pergunta}")
        resposta = responder(pergunta, db_global)
        historico.append((pergunta, resposta))
        return historico, historico
        
    except Exception as e:
        resposta = f"❌ Erro ao gerar resposta: {str(e)}"
        historico.append((pergunta, resposta))
        return historico, historico

# Criar interface Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Chatbot") as demo:
    
    gr.Markdown("# 🤖 Chatbot RAG com Ollama")
    gr.Markdown("Faça perguntas sobre os documentos da base de conhecimento")
    
    # Área de chat
    chatbot = gr.Chatbot(
        label="Conversa",
        height=500,
        placeholder="Faça sua primeira pergunta sobre os documentos..."
    )
    
    # Barra de pergunta e botão
    with gr.Row():
        pergunta_input = gr.Textbox(
            label="",
            placeholder="Digite sua pergunta aqui...",
            scale=5,
            show_label=False
        )
        btn_pesquisar = gr.Button("🔍 Pesquisar", variant="primary", scale=1, size="lg")
    
    with gr.Row():
        btn_limpar = gr.Button("🗑️ Limpar Histórico", variant="secondary", size="sm")
    
    gr.Markdown("---")
    gr.Markdown("""
    **💡 Informações:**
    - Modelo: Llama3 (via Ollama)
    - Embeddings: nomic-embed-text
    - Banco: Chroma Vector Store
    """)
    
    # Eventos
    btn_pesquisar.click(
        fn=responder_pergunta,
        inputs=[pergunta_input, chatbot],
        outputs=[chatbot, chatbot]
    ).then(
        lambda: "",
        outputs=pergunta_input
    )
    
    pergunta_input.submit(
        fn=responder_pergunta,
        inputs=[pergunta_input, chatbot],
        outputs=[chatbot, chatbot]
    ).then(
        lambda: "",
        outputs=pergunta_input
    )
    
    btn_limpar.click(
        lambda: [],
        outputs=chatbot
    )

# Iniciar a interface
if __name__ == "__main__":
    if db_global is None:
        print("\n⚠️  ATENÇÃO: Banco de dados não foi carregado!")
        print("Execute 'python create_db.py' primeiro para criar o banco.\n")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860
    )