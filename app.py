import gradio as gr
import os
import time
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

CAMINHO_DB = "chroma"

# Embeddings e LLM
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3.2", temperature=0.1)

# Exemplos few-shot
FEW_SHOT = """
Exemplos de perguntas e respostas:

Pergunta: O que é o PIBIC?
Resposta: O PIBIC é um programa nacional de iniciação científica voltado a alunos de graduação.

Pergunta: Qual o valor da bolsa do PIBIC?
Resposta: O valor atual da bolsa é R$ 700 por mês.

Pergunta: Quem pode participar?
Resposta: Estudantes regularmente matriculados, com bom desempenho acadêmico e sem vínculo empregatício.
"""

# Template do prompt
PROMPT_TEMPLATE = """
Você é um assistente especializado em responder perguntas sobre EDITAIS e PROGRAMAS DE INICIAÇÃO CIENTÍFICA.

{few_shot}

Responda APENAS com base no contexto abaixo.

Contexto:
{contexto}

Pergunta: {pergunta}

Regras:
1. Só use informações que estejam no contexto.
2. Seja claro e objetivo.
3. Se a resposta exata não estiver no texto, diga o que existe de relevante.
4. Se nada existir, responda: "Informação não encontrada nos documentos."

Resposta:
"""

# -----------------------------
# CARREGAR BANCO VETORIAL
# -----------------------------
def carregar_db():
    if not os.path.exists(CAMINHO_DB):
        print("❌ Banco vetorial não encontrado. Execute create_db.py primeiro.")
        return None

    try:
        db = Chroma(
            persist_directory=CAMINHO_DB,
            embedding_function=embeddings,
            collection_name="db1"
        )
        print("✅ Banco vetorial carregado")
        return db
    except Exception as e:
        print("❌ Erro ao carregar banco:", e)
        return None


db_global = carregar_db()

# -----------------------------
# RAG — BUSCA + GERAÇÃO
# -----------------------------
def responder(pergunta, db):

    docs_semantic = db.similarity_search(pergunta, k=6)

    if not docs_semantic:
        return "❌ Nenhum documento relevante encontrado."

    keywords = [w.lower() for w in pergunta.split() if len(w) > 3]

    docs_final = {}
    for d in docs_semantic:
        doc_id = hash(d.page_content[:200])
        docs_final[doc_id] = d

    docs_sorted = list(docs_final.values())
    docs_sorted.sort(
        key=lambda d: sum(k in d.page_content.lower() for k in keywords),
        reverse=True
    )

    contexto = ""
    limite = 4500

    for d in docs_sorted:
        trecho = d.page_content[:1800]
        if len(contexto) + len(trecho) <= limite:
            contexto += "\n\n" + trecho
        else:
            break

    prompt = PROMPT_TEMPLATE.format(
        few_shot=FEW_SHOT,
        contexto=contexto,
        pergunta=pergunta
    )

    resposta_final = ""
    for chunk in llm.stream(prompt):
        resposta_final += chunk

    return resposta_final

# -----------------------------
# FUNÇÃO PARA O CHATBOT (GRADIO 4.x)
# -----------------------------
def chat_function(pergunta, chat_history):
    if chat_history is None:
        chat_history = []

    # Adiciona a mensagem do usuário
    chat_history.append({
        "role": "user",
        "content": pergunta
    })

    # Tratar erros
    if not pergunta.strip():
        resposta = "⚠️ Digite uma pergunta válida."
    elif db_global is None:
        resposta = "❌ Banco de dados não está carregado. Execute create_db.py primeiro."
    else:
        inicio = time.time()
        resposta = responder(pergunta, db_global)
        tempo = time.time() - inicio
        resposta += f"\n\n---\n⏱️ **Tempo de resposta:** {tempo:.2f} segundos"

    # Adiciona resposta do assistente no formato correto
    chat_history.append({
        "role": "assistant",
        "content": resposta
    })

    return chat_history



# -----------------------------
# INTERFACE GRADIO
# -----------------------------
with gr.Blocks(title="RAG Chatbot") as demo:

    gr.Markdown("""
    # 🤖 Chatbot RAG — Editais & Iniciação Científica  
    Pergunte qualquer coisa sobre os documentos carregados.
    """)

    chatbot = gr.Chatbot(label="Chat", height=500)



    pergunta = gr.Textbox(
        placeholder="Digite sua pergunta...",
        label="Sua pergunta"
    )

    enviar = gr.Button("🔍 Pesquisar", variant="primary")
    limpar = gr.Button("🗑️ Limpar Histórico")

    enviar.click(
        chat_function,
        inputs=[pergunta, chatbot],
        outputs=chatbot
    )

    pergunta.submit(
        chat_function,
        inputs=[pergunta, chatbot],
        outputs=chatbot
    )

    limpar.click(lambda: [], outputs=chatbot)


# Executar servidor
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )
