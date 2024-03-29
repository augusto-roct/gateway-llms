from fastapi import APIRouter, Request, Response
from gateway_llms.app.interfaces.embeddings import TextEmbedding
from gateway_llms.app.modules.chat.embeddings import get_embeddings

from gateway_llms.app.utils.logs import LogApplication


router = APIRouter()


@router.post(
    path="/indexing",
    description="Transforme o seu texto em um vetor de indices"
)
async def text_to_embedding(
    request: Request,
    response: Response,
    text_embedding: TextEmbedding
):
    log_user = LogApplication(request, await request.body())

    data = await get_embeddings([text_embedding.text], log_user)

    data = {"data": list(data[0])}

    return data
