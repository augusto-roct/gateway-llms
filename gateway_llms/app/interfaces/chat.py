from pydantic import BaseModel, Field


class ChatLLMCompletion(BaseModel):
    text: str = Field(
        ...,
        description="Entrada do usuário para conversar com o modelo"
    )
    model: str = Field(
        None,
        description="Modelo que será utilizado na conversa"
    )
