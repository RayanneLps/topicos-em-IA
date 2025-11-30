# Melhorias V2 - Foco em Precisão e Few-Shot Learning

## Problema Identificado
As otimizações anteriores reduziram o tempo, mas também reduziram drasticamente a precisão das respostas (muitas "não encontradas").

## Soluções Implementadas

### 1. **Ajuste de Parâmetros de Busca** ✅

#### Antes (muito restritivo):
- `k=3` documentos
- Filtro de score < 0.85
- Contexto máximo: 3000 caracteres
- 1500 caracteres por documento

#### Agora (balanceado):
- `k=7` documentos (busca semântica inicial)
- **Sem filtro de score** (aceita todos os documentos encontrados)
- Contexto máximo: **5000 caracteres** (aumentado)
- **2000 caracteres por documento** (aumentado)
- Até **6 documentos** mais relevantes no contexto final

### 2. **Busca Híbrida** ✅

Implementada combinação de:
- **Busca semântica**: Usa embeddings para encontrar documentos similares
- **Busca por palavras-chave**: Prioriza documentos que contêm palavras-chave da pergunta
- **Remoção de duplicatas**: Evita repetir o mesmo conteúdo

**Benefício**: Melhor cobertura e maior chance de encontrar informações relevantes.

### 3. **Few-Shot Learning** ✅

Adicionados exemplos no prompt para guiar o modelo:

```
Exemplos de perguntas e respostas corretas:
- O que é o PIBIC?
- Qual o valor da bolsa do PIBIC?
- Quais são os requisitos para participar?
```

**Benefício**: O modelo aprende o formato e estilo de resposta esperado.

### 4. **Prompt Melhorado** ✅

Novo prompt inclui:
- Instruções claras sobre quando dizer "não encontrado"
- Orientação para mencionar informações relacionadas mesmo sem resposta exata
- Foco em ser direto e objetivo

### 5. **Pré-processamento Menos Agressivo** ✅

Ajustes no `preprocessar_pdfs.py`:
- **Preserva mais informação**: Não remove linhas curtas automaticamente
- **Mantém emails**: Podem ser informações importantes
- **Limite reduzido**: De 50 para 20 caracteres mínimos por documento
- **Limpeza mais conservadora**: Remove apenas conteúdo claramente irrelevante

## Comparação de Estratégias

### ❌ Estratégia Anterior (rápida mas imprecisa):
```
k=3 → Filtro score → Contexto 3000 → Resposta rápida mas muitas "não encontradas"
```

### ✅ Estratégia Atual (balanceada):
```
k=7 → Busca híbrida → Contexto 5000 → Few-shot → Resposta mais precisa
```

## Sobre Transformar PDFs em Planilhas

**Não recomendado** para este caso porque:
- Editais são documentos textuais estruturados, não dados tabulares
- Planilhas perderiam a estrutura hierárquica (seções, subseções)
- Relacionamentos entre informações seriam perdidos
- Embeddings funcionam melhor com texto contínuo

**Alternativa melhor**: Manter PDFs mas melhorar a extração e chunking.

## Outras Otimizações Recomendadas (Futuro)

### 1. **Re-ranking de Documentos**
- Usar modelo de re-ranking para ordenar documentos por relevância
- Exemplo: `cross-encoder/ms-marco-MiniLM`

### 2. **Chunking Inteligente**
- Chunks baseados em estrutura do documento (seções, parágrafos)
- Manter contexto entre chunks relacionados

### 3. **Cache de Embeddings**
- Cachear embeddings de perguntas frequentes
- Reduzir tempo de busca

### 4. **Compressão de Contexto**
- Resumir chunks muito longos antes de enviar ao LLM
- Manter apenas informações mais relevantes

### 5. **Modelo Mais Rápido**
- Considerar modelos menores para respostas rápidas
- Exemplo: `llama3.2:1b` ou `gemma2:2b` para respostas simples

## Próximos Passos

1. **Recriar o banco de dados**:
   ```bash
   python create_db.py
   ```

2. **Testar as melhorias**:
   ```bash
   python testar_perguntas.py
   ```

3. **Ajustar parâmetros se necessário**:
   - Se ainda houver muitas "não encontradas": aumentar `k` ou contexto
   - Se tempo estiver muito alto: reduzir contexto ou número de docs

## Resultados Esperados

- ✅ **Precisão**: 70-90% de respostas encontradas (antes: ~30%)
- ⏱️ **Tempo**: 80-150 segundos (ainda melhor que os 200-280s originais)
- 📊 **Qualidade**: Respostas mais completas e relevantes

