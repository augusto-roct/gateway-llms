from typing import List
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
    model: str = Field(
        None,
        description="Modelo que será utilizado na conversa"
    )
    messages: List[ChatMessages] = Field(
        None,
        description="Histórico de conversa com o modelo"
    )
