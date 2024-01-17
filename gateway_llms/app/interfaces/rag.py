from pydantic import BaseModel, Field

from gateway_llms.app.interfaces.chat import ChatConfig


class RagConfig(BaseModel):
    model_name: str = Field(
        ...,
        description="Nome do modelo que será utilizado"
    )
    chunk_size: int = Field(
        None,
        description="Tamanho do segmento de cada vetor de indices"
    )
    chunk_overlap: int = Field(
        None,
        description="Tamanho da sobreposição no segmento"
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
