import re
import time
import os
from datetime import datetime
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

CAMINHO_DB = "chroma"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3")

# Few-shot examples para melhorar a qualidade das respostas
FEW_SHOT_EXAMPLES = """
Exemplos de perguntas e respostas corretas:

Pergunta: O que é o PIBIC?
Resposta: O PIBIC (Programa Institucional de Bolsas de Iniciação Científica) é um programa que visa despertar a vocação científica e incentivar talentos potenciais entre estudantes de graduação.

Pergunta: Qual o valor da bolsa do PIBIC?
Resposta: O valor da bolsa do PIBIC é de R$ 700,00 mensais.

Pergunta: Quais são os requisitos para participar?
Resposta: Os requisitos incluem estar regularmente matriculado em curso de graduação, ter bom desempenho acadêmico e não possuir vínculo empregatício.
"""

PROMPT_TEMPLATE = """
Você é um assistente especializado em responder perguntas sobre editais de programas de iniciação científica.

{Few_shot_examples}

Agora, responda à pergunta abaixo usando APENAS as informações do contexto fornecido.

Contexto: {contexto}

Pergunta: {pergunta}

Instruções:
1. Use APENAS informações do contexto fornecido
2. Seja direto e objetivo na resposta
3. Se a informação estiver no contexto, responda de forma completa
4. Se não encontrar a resposta exata, mas houver informação relacionada, mencione o que encontrou
5. Apenas diga "Informação não encontrada nos documentos" se realmente não houver NENHUMA informação relacionada

Resposta:
"""

def carregar_db():
    if not os.path.exists(CAMINHO_DB):
        print(f"ERRO: Banco de dados '{CAMINHO_DB}' nao encontrado!")
        print("Execute primeiro 'create_db.py' para criar o banco.")
        return None
    
    db = Chroma(
        persist_directory=CAMINHO_DB,
        embedding_function=embeddings,
        collection_name="db1"
    )
    print("OK: Banco de dados carregado com sucesso!")
    return db

def responder(pergunta, db):
    # Busca melhorada: usar mais documentos para melhor cobertura
    # Primeiro tenta busca semântica
    docs_semanticos = db.similarity_search(pergunta, k=7)
    
    if not docs_semanticos:
        return "Nenhum documento relevante encontrado."
    
    # Busca também por palavras-chave importantes (busca híbrida)
    palavras_chave = [palavra.lower() for palavra in pergunta.split() if len(palavra) > 3]
    
    # Combinar resultados (remover duplicatas)
    todos_docs = {}
    for doc in docs_semanticos:
        # Usar hash do conteúdo como chave para evitar duplicatas
        doc_id = hash(doc.page_content[:100])
        if doc_id not in todos_docs:
            todos_docs[doc_id] = doc
    
    # Ordenar por relevância (documentos que contêm palavras-chave têm prioridade)
    docs_ordenados = list(todos_docs.values())
    docs_ordenados.sort(key=lambda d: sum(1 for palavra in palavras_chave if palavra in d.page_content.lower()), reverse=True)
    
    # Construir contexto otimizado (aumentar limite para melhor cobertura)
    contexto_limitado = []
    tamanho_maximo_contexto = 5000  # Aumentado para melhor cobertura
    
    for doc in docs_ordenados[:6]:  # Pegar até 6 documentos mais relevantes
        texto = doc.page_content[:2000]  # Aumentado de 1500 para 2000
        contexto_atual = "\n\n".join(contexto_limitado)
        novo_contexto = contexto_atual + "\n\n" + texto if contexto_atual else texto
        
        if len(novo_contexto) <= tamanho_maximo_contexto:
            contexto_limitado.append(texto)
        else:
            # Adicionar o que couber
            espaco_restante = tamanho_maximo_contexto - len(contexto_atual)
            if espaco_restante > 500:  # Só adiciona se houver espaço significativo
                contexto_limitado.append(texto[:espaco_restante-10])
            break
    
    contexto = "\n\n".join(contexto_limitado)
    
    # Prompt com few-shot learning
    prompt = PROMPT_TEMPLATE.format(
        Few_shot_examples=FEW_SHOT_EXAMPLES,
        contexto=contexto, 
        pergunta=pergunta
    )

    resposta = llm.invoke(prompt)
    return resposta

def extrair_perguntas(arquivo):
    """Extrai as perguntas do arquivo de teste"""
    perguntas = []
    with open(arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    
    # Padrão para encontrar perguntas numeradas
    # Exemplo: "1. O que é o PIBIC e qual o seu objetivo? [Nível: Fácil]"
    padrao = r'(\d+)\.\s+(.+?)\s+\[Nível:'
    matches = re.findall(padrao, conteudo)
    
    for num, pergunta in matches:
        perguntas.append((int(num), pergunta.strip()))
    
    return perguntas

def analisar_resposta(resposta):
    """Analisa se a resposta foi encontrada ou não"""
    resposta_lower = resposta.lower()
    
    if "não encontrada" in resposta_lower or "não encontrado" in resposta_lower:
        return "não encontrou a resposta"
    elif "informação não encontrada" in resposta_lower:
        return "não encontrou a resposta"
    elif len(resposta.strip()) < 50:  # Resposta muito curta pode indicar problema
        return "resposta muito curta"
    else:
        return "respondeu corretamente"

def gerar_relatorio(perguntas, resultados, nome_arquivo="relatorio_de_teste.txt"):
    """Gera o relatório no formato especificado"""
    data_atual = datetime.now().strftime("%d\\%m\\%Y")
    
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        f.write(f"teste 1 : ({data_atual}):\n")
        f.write("    chunk_size=800,\n")
        f.write("    chunk_overlap=200,\n")
        f.write("    length_function=len,\n")
        f.write("    add_start_index=True, #versao funcional 1\n")
        f.write("    separators=[\"\\n\\n\", \"\\n\", \". \", \" \", \"\"]\n")
        f.write("\n")
        
        for num, tempo, status in resultados:
            tempo_int = int(tempo)
            f.write(f"    pergunta {num}:{tempo_int}s e {status}\n")
        
        f.write("\n")
        f.write("modelos:\n")
        # Adiciona informações dos modelos (você pode ajustar isso)
        f.write("llama3:latest\n")
        f.write("nomic-embed-text:latest\n")

def main():
    print("Inicializando sistema...")
    db = carregar_db()
    
    if db is None:
        print("ERRO: Nao foi possivel carregar o banco de dados!")
        return
    
    print("\nLendo perguntas do arquivo...")
    perguntas = extrair_perguntas("perguntas_teste_rag.txt")
    
    if not perguntas:
        print("ERRO: Nenhuma pergunta encontrada no arquivo!")
        return
    
    print(f"OK: {len(perguntas)} perguntas encontradas\n")
    
    resultados = []
    
    for num, pergunta in perguntas:
        print(f"Processando pergunta {num}: {pergunta[:50]}...")
        
        try:
            inicio = time.time()
            resposta = responder(pergunta, db)
            tempo_decorrido = time.time() - inicio
            
            status = analisar_resposta(resposta)
            
            resultados.append((num, tempo_decorrido, status))
            
            print(f"   Tempo: {tempo_decorrido:.2f}s - Status: {status}")
            print(f"   Resposta: {resposta[:100]}...\n")
            
        except Exception as e:
            print(f"   ERRO: {str(e)}\n")
            resultados.append((num, 0, f"erro: {str(e)[:30]}"))
    
    print("\nGerando relatorio...")
    gerar_relatorio(perguntas, resultados)
    print("OK: Relatorio gerado com sucesso em 'relatorio_de_teste.txt'!")

if __name__ == "__main__":
    main()

