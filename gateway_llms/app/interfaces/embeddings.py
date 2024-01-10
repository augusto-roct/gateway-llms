from pydantic import BaseModel, Field


class TextEmbedding(BaseModel):
    text: str = Field(
        ...,
        description="Entrada do usuário para transformar o texto em um vetor de indices"
    )
