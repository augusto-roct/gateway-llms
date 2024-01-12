from pydantic import BaseModel, Field


class RagQuery(BaseModel):
    text: str = Field(
        ...,
        description="Entrada do usuário para transformar o texto em um vetor de indices"
    )
    document_name: str = Field(
        ...,
        description="O nome do documento que será utilizado como contexto para conversa"
    )
