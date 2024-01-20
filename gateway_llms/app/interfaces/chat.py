from typing import Iterable, List
from pydantic import BaseModel, Field


class SafetySettings(BaseModel):
    HARM_CATEGORY_HARASSMENT: str = Field(
        "BLOCK_NONE",
        description="Configuração para respostas de assédio"
    )
    HARM_CATEGORY_HATE_SPEECH: str = Field(
        "BLOCK_NONE",
        description="Configuração para respostas de discurso de ódio"
    )
    HARM_CATEGORY_SEXUALLY_EXPLICIT: str = Field(
        "BLOCK_NONE",
        description="Configuração para respostas com conteúdo sexualmente explícito"
    )
    HARM_CATEGORY_DANGEROUS_CONTENT: str = Field(
        "BLOCK_NONE",
        description="Configuração para respostas com conteúdos perigosos"
    )


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
        0.3
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
    generation_config: ChatConfig = Field(
        ...,
        description="Configuração que será utilizada pelo modelo"
    )
    safety_settings: SafetySettings = Field(
        ...,
        description="Configuração para respostas apropiadas"
    )
