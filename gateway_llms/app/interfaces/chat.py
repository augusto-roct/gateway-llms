from typing import Iterable, List
from pydantic import BaseModel, Field


class ChatMessages(BaseModel):
    role: str = Field(
        ...,
        description="Identifica quem escreveu a mensagem atual"
    )
    content: str = Field(
        ...,
        description="Conteúdo da mensagem"
    )


class ChatConfig(BaseModel):
    model_name: str = Field(
        ...,
        description="Nome do modelo que será utilizado"
    )
    candidate_count: int = Field(
        1
    )
    stop_sequences: Iterable[str] = Field(
        None
    )
    max_output_tokens: int = Field(
        None
    )
    temperature: float = Field(
        None
    )
    top_p: float = Field(
        None
    )
    top_k: int = Field(
        None
    )


class ChatLLMCompletion(BaseModel):
    text: str = Field(
        ...,
        description="Entrada do usuário para conversar com o modelo"
    )
    system: str = Field(
        None,
        description="Prompt utilizado pelo sistema para definir o "
        "comportamento do modelo"
    )
    messages: List[ChatMessages] = Field(
        None,
        description="Histórico de conversa com o modelo"
    )
    parameters: dict = Field(
        None,
        description="Parâmetros utilizados no prompt"
    )
    config: ChatConfig = Field(
        None,
        description="Configuração que será utilizada pelo modelo"
    )
