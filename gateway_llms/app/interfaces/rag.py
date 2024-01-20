from pydantic import BaseModel, Field

from gateway_llms.app.interfaces.chat import ChatConfig


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
    chat_model_name: str = Field(
        ...,
        description="Nome do modelo, para chat, que será utilizado"
    )
    embedding_model_name: str = Field(
        ...,
        description="Nome do modelo, para embedding, que será utilizado"
    )
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
    config: RagConfig = Field(
        ...,
        description="Configuração que será utilizada pelo modelo"
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
    config: ChatConfig = Field(
        None,
        description="Configuração que será utilizada pelo modelo"
    )
