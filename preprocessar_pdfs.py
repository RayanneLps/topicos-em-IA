"""
Módulo de pré-processamento de PDFs para otimizar o RAG
- Limpeza de texto
- Normalização
- Remoção de conteúdo irrelevante
- Otimização de estrutura
"""

import re
import os
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document


def limpar_texto(texto: str) -> str:
    """
    Remove espaços extras, quebras de linha desnecessárias e normaliza o texto
    Versão menos agressiva para preservar mais informação
    """
    if not texto:
        return ""
    
    # Remover caracteres de controle (exceto quebras de linha e tabs)
    texto = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', texto)
    
    # Normalizar múltiplos espaços em branco (mas manter quebras de linha)
    texto = re.sub(r'[ \t]+', ' ', texto)  # Apenas espaços e tabs, não quebras de linha
    
    # Remover quebras de linha excessivas (mais de 3 seguidas)
    texto = re.sub(r'\n{4,}', '\n\n\n', texto)
    
    # Normalizar espaços antes de pontuação (mas não remover quebras importantes)
    texto = re.sub(r' +([.,;:!?])', r'\1', texto)
    
    # Remover espaços no início e fim
    texto = texto.strip()
    
    return texto


def remover_conteudo_irrelevante(texto: str) -> str:
    """
    Remove apenas conteúdo claramente irrelevante (versão menos agressiva)
    """
    if not texto:
        return ""
    
    # Remover apenas números de página isolados em linhas próprias (mais conservador)
    # Só remove se for apenas um número de 1-3 dígitos sozinho
    texto = re.sub(r'^\s*\d{1,3}\s*$', '', texto, flags=re.MULTILINE)
    
    # Remover rodapés comuns de paginação (mas preservar outros números)
    texto = re.sub(r'P[áa]gina\s+\d+\s+de\s+\d+', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'Page\s+\d+\s+of\s+\d+', '', texto, flags=re.IGNORECASE)
    
    # Remover URLs muito longas (mas preservar referências curtas)
    texto = re.sub(r'https?://[^\s]{50,}', '', texto)
    
    # NÃO remover emails (podem ser importantes)
    # NÃO remover linhas curtas (podem conter informação importante)
    
    return texto


def normalizar_quebras(texto: str) -> str:
    """
    Normaliza quebras de linha para melhor estruturação
    """
    if not texto:
        return ""
    
    # Quebras de parágrafo (duas ou mais quebras de linha)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    
    # Manter quebras após pontuação final
    texto = re.sub(r'([.!?])\s*\n', r'\1\n\n', texto)
    
    return texto.strip()


def extrair_metadados_uteis(doc: Document) -> dict:
    """
    Extrai e estrutura metadados úteis do documento
    """
    metadados = doc.metadata.copy() if doc.metadata else {}
    
    # Adicionar informações úteis se não existirem
    if 'source' in metadados:
        # Extrair nome do arquivo
        nome_arquivo = os.path.basename(metadados['source'])
        metadados['nome_arquivo'] = nome_arquivo
        
        # Tentar identificar tipo de documento pelo nome
        if 'PIBIC' in nome_arquivo.upper():
            metadados['tipo_documento'] = 'PIBIC'
        elif 'PIBITI' in nome_arquivo.upper():
            metadados['tipo_documento'] = 'PIBITI'
        elif 'ICV' in nome_arquivo.upper():
            metadados['tipo_documento'] = 'ICV'
        elif 'ADITIVO' in nome_arquivo.upper():
            metadados['tipo_documento'] = 'ADITIVO'
    
    return metadados


def preprocessar_documento(doc: Document) -> Document:
    """
    Aplica todas as transformações de pré-processamento em um documento
    """
    texto_original = doc.page_content
    
    # Aplicar transformações
    texto = limpar_texto(texto_original)
    texto = remover_conteudo_irrelevante(texto)
    texto = normalizar_quebras(texto)
    
    # Extrair metadados úteis
    metadados = extrair_metadados_uteis(doc)
    
    # Criar novo documento processado
    doc_processado = Document(
        page_content=texto,
        metadata=metadados
    )
    
    return doc_processado


def preprocessar_documentos(documentos: List[Document]) -> List[Document]:
    """
    Pré-processa uma lista de documentos
    """
    documentos_processados = []
    
    print(f"Preprocessando {len(documentos)} documentos...")
    
    for i, doc in enumerate(documentos, 1):
        try:
            doc_processado = preprocessar_documento(doc)
            
            # Só adicionar se o documento tiver conteúdo relevante (reduzido de 50 para 20 caracteres)
            if doc_processado.page_content.strip() and len(doc_processado.page_content.strip()) > 20:
                documentos_processados.append(doc_processado)
            
            if i % 10 == 0:
                print(f"  Processados: {i}/{len(documentos)}")
                
        except Exception as e:
            print(f"  Erro ao processar documento {i}: {e}")
            continue
    
    print(f"OK: {len(documentos_processados)} documentos processados com sucesso")
    print(f"   ({len(documentos) - len(documentos_processados)} documentos removidos por falta de conteúdo)")
    
    return documentos_processados


def carregar_e_preprocessar_pdfs(pasta_base: str = "base") -> List[Document]:
    """
    Carrega PDFs de uma pasta e aplica pré-processamento
    """
    if not os.path.exists(pasta_base):
        print(f"ERRO: Pasta '{pasta_base}' nao encontrada!")
        return []
    
    print(f"Carregando PDFs de '{pasta_base}'...")
    loader = PyPDFDirectoryLoader(pasta_base, glob="**/*.pdf")
    documentos = loader.load()
    
    if not documentos:
        print("ERRO: Nenhum documento carregado!")
        return []
    
    print(f"OK: {len(documentos)} documentos carregados")
    
    # Aplicar pré-processamento
    documentos_processados = preprocessar_documentos(documentos)
    
    return documentos_processados


if __name__ == "__main__":
    # Teste do pré-processamento
    docs = carregar_e_preprocessar_pdfs("base")
    print(f"\nTotal de documentos processados: {len(docs)}")
    
    if docs:
        print(f"\nExemplo de documento processado:")
        print(f"  Fonte: {docs[0].metadata.get('source', 'N/A')}")
        print(f"  Tamanho original: {len(docs[0].page_content)} caracteres")
        print(f"  Primeiros 200 caracteres: {docs[0].page_content[:200]}...")

