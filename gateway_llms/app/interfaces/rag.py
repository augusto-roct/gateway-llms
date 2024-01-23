from pydantic import BaseModel, Field

from gateway_llms.app.interfaces.chat import ChatConfig, SafetySettings


class Bm25Retriever(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve utilizar o algoritmo bm25"
    )
    similarity_top_k: int = Field(
        5,
        description="Número de nós a serem utilizados"
    )


class RagRetriever(BaseModel):
    bm25_retriever: Bm25Retriever = Field(
        ...,
        description="Utiliza o algoritmo bm25"
    )


class SimilarityPostprocessors(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve utilizar similaridade no pós-processamento"
    )
    similarity_cutoff: float = Field(
        0.7,
        description="Limiar que será utilzado"
    )


class KeywordPostprocessors(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve utilizar palavras-chaves no pós-processamento"
    )
    required_keywords: list[str] = Field(
        [],
        description="Lista de palavras-chaves necessárias"
    )
    exclude_keywords: list[str] = Field(
        [],
        description="Lista de palavras-chaves excluidas"
    )
    lang: str = Field(
        "pt",
        description="Linguagem utilizada"
    )


class MetadataReplacementPostProcessor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve utilizar a substituição do conteúdo por metadado no pós-processamento"
    )
    target_metadata_key: str = Field(
        "window",
        description="Indica qual metadado será utilizado"
    )


class LongContextReorderPostprocessors(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve reorganizar os nós no pós-processamento"
    )


class SentenceEmbeddingOptimizerPostProcessor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve utilizar o vetor de indices para remover frases não relevantes para pergunta no pós-processamento"
    )
    percentile_cutoff: float = Field(
        0.5,
        description="Indica qual o percentual de relevância da frase"
    )
    threshold_cutoff: float = Field(
        0.7,
        description="Limiar que será utilzado"
    )


class SentenceTransformerRerankPostProcessor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve utilizar o vetor de indices para remover frases não relevantes para pergunta no pós-processamento"
    )
    top_n: int = Field(
        3,
        description="Número de nós a serem retornados"
    )


class LLMRerankPostprocessors(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve utilizar um LLM para reordenar nós solicitando que o LLM retorne "
        "os documentos relevantes e uma pontuação de quão relevantes eles são no pós-processamento"
    )
    top_n: int = Field(
        2,
        description="Número de nós a serem retornados"
    )


class FixedRecencyPostprocessor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve retornar os nós K superiores classificados por data no pós-processamento"
    )
    tok_k: int = Field(
        2,
        description="Número de nós a serem retornados"
    )


class EmbeddingRecencyPostprocessor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve retornar os nós K superiores classificados por data e remover nós mais "
        "antigos que são muito semelhantes no pós-processamento"
    )
    similarity_cutoff: float = Field(
        0.7,
        description="Limiar que será utilzado"
    )


class TimeWeightedPostprocessor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve retornar os nós K superiores aplicando uma reclassificação ponderada "
        "pelo tempo a cada nó no pós-processamento"
    )
    time_decay: float = Field(
        0.99,
        description="Limiar que será utilzado"
    )
    top_k: int = Field(
        1,
        description="Número de nós a serem retornados"
    )


class PrevNextNodePostprocessor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve ler as relações de nó e buscar todos os nós que vêm anteriormente, "
        "em seguida ou em ambos no pós-processamento"
    )
    num_nodes: int = Field(
        1,
        description="Número de nós"
    )
    mode: str = Field(
        "next",
        description="Modo a ser utilizado aceita apenas 'next', 'previous', ou 'both'"
    )


class AutoPrevNextNodePostprocessor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve ler as relações de nó e buscar todos os nós que vêm anteriormente, "
        "em seguida ou em ambos (selecionado automaticamente usando o LLM) no pós-processamento"
    )
    num_nodes: int = Field(
        1,
        description="Número de nós"
    )


class RankGPTRerankPostprocessor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve usar o agente RankGPT para reclassificar documentos de acordo com a relevância no pós-processamento"
    )
    top_n: int = Field(
        3,
        description="Número de nós"
    )


class NodePostprocessors(BaseModel):
    similarity: SimilarityPostprocessors = Field(
        ...,
        description="Utiliza similaridade no pós-processamento"
    )
    keyword: KeywordPostprocessors = Field(
        ...,
        description="Utiliza palavras chaves no pós-processamento"
    )
    metadata_replacement: MetadataReplacementPostProcessor = Field(
        ...,
        description="Utiliza a substituição do conteúdo por metadado no pós-processamento"
    )
    log_context_reorder: LongContextReorderPostprocessors = Field(
        ...,
        description="Reorganiza os nós no pós-processamento"
    )
    sentence_embedding_optimizer: SentenceEmbeddingOptimizerPostProcessor = Field(
        ...,
        description="Utiliza o vetor de indices para remover frases não relevantes para pergunta"
    )
    sentence_transformer_rerank: SentenceTransformerRerankPostProcessor = Field(
        ...,
        description="Utiliza os codificadores cruzados do pacote de transformador de sentença para reordenar "
        "nós e retorna os nós N superiores"
    )
    llm_rerank: LLMRerankPostprocessors = Field(
        ...,
        description="Utiliza um LLM para reordenar nós solicitando que o LLM retorne os documentos relevantes "
        "e uma pontuação de quão relevantes eles são"
    )
    fix_recency: FixedRecencyPostprocessor = Field(
        ...,
        description="Utiliza os nós K superiores classificados por data"
    )
    embedding_recency: EmbeddingRecencyPostprocessor = Field(
        ...,
        description="Utiliza os nós K superiores classificados por data e remover nós mais antigos que são muito semelhantes"
    )
    time_weight: TimeWeightedPostprocessor = Field(
        ...,
        description="Utiliza os os nós K superiores aplicando uma reclassificação ponderada pelo tempo a cada nó"
    )
    prev_next_node: PrevNextNodePostprocessor = Field(
        ...,
        description="Utiliza as relações de nó e busca todos os nós que vêm anteriormente, em seguida ou em ambos"
    )
    auto_prev_next_node: AutoPrevNextNodePostprocessor = Field(
        ...,
        description="Utiliza as relações de nó e busca todos os nós que vêm anteriormente, em seguida ou em "
        "ambos (selecionado automaticamente usando o LLM)"
    )
    rank_gpt_rerank: RankGPTRerankPostprocessor = Field(
        ...,
        description="Utiliza o agente RankGPT para reclassificar documentos de acordo com a relevância"
    )


class TitleExtractor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve extrair titulos dos segmentos"
    )
    quantity: int = Field(
        5,
        description="Quantidade de titulos a serem extraidos"
    )


class QuestionAnsweredExtractor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve extrair questões e respostas dos segmentos"
    )
    quantity: int = Field(
        5,
        description="Quantidade de questões a serem extraidas"
    )


class KeywordExtractor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve extrair palavras chaves dos segmentos"
    )
    quantity: int = Field(
        5,
        description="Quantidade de palavras chaves a serem extraidas"
    )


class SummaryExtractor(BaseModel):
    is_use: bool = Field(
        False,
        description="Indica se deve extrair resumos dos segmentos"
    )
    types: list[str] = Field(
        ['self'],
        description="Como deve ser feito o resumo. Aceita somente 'self', 'prev', 'next'"
    )


class RagExtractors(BaseModel):
    title: TitleExtractor = Field(
        None,
        description="Extrai titulos dos segmentos"
    )
    questions: QuestionAnsweredExtractor = Field(
        None,
        description="Extrai questões e respostas dos segmentos"
    )
    keywords: KeywordExtractor = Field(
        None,
        description="Extrai palavras chaves dos segmentos"
    )
    summary: SummaryExtractor = Field(
        None,
        description="Extrai resumos dos segmentos"
    )


class RagConfig(BaseModel):
    chunk_size: int = Field(
        None,
        description="Tamanho do segmento de cada vetor de indices"
    )
    chunk_overlap: int = Field(
        None,
        description="Tamanho da sobreposição no segmento"
    )
    extractors: RagExtractors = Field(
        ...,
        description="Configura a extração de metadados no documento"
    )


class RagDocument(BaseModel):
    name: str = Field(
        None,
        description="Nome do documento que será transformado em vetores de indices"
    )
    chat_model_name: str = Field(
        ...,
        description="Nome do modelo, para chat, que será utilizado"
    )
    embedding_model_name: str = Field(
        ...,
        description="Nome do modelo, para embedding, que será utilizado"
    )
    config: RagConfig = Field(
        ...,
        description="Configuração que será utilizada pelo modelo para RAG"
    )
    generation_config: ChatConfig = Field(
        ...,
        description="Configuração que será utilizada pelo modelo para Chat"
    )
    safety_settings: SafetySettings = Field(
        ...,
        description="Configuração para respostas apropiadas"
    )


class RagQuery(BaseModel):
    text: str = Field(
        ...,
        description="Entrada do usuário para transformar o texto em um vetor de indices"
    )
    document_name: str = Field(
        ...,
        description="O nome do documento que será utilizado como contexto para conversa"
    )
    system_prompt: str = Field(
        None,
        description="Prompt utilizado pelo sistema para definir o "
        "comportamento do modelo"
    )
    chat_model_name: str = Field(
        ...,
        description="Nome do modelo, para chat, que será utilizado"
    )
    embedding_model_name: str = Field(
        ...,
        description="Nome do modelo, para embedding, que será utilizado"
    )
    generation_config: ChatConfig = Field(
        ...,
        description="Configuração que será utilizada pelo modelo"
    )
    similarity_top_k: int = Field(
        3,
        description="Número de nós que serão utilizados"
    )
    retriever: RagRetriever = Field(
        ...,
        description="Configuração para recuperação de documentos"
    )
    node_postprocessors: NodePostprocessors = Field(
        ...,
        description="Configuração de pós processamento que será utilizada pelo modelo"
    )
    safety_settings: SafetySettings = Field(
        ...,
        description="Configuração para respostas apropiadas"
    )
