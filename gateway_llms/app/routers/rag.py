from fastapi import APIRouter, Request, Response, File, UploadFile
from gateway_llms.app.controllers.rag import document_to_embeddings, storage_to_query
from gateway_llms.app.interfaces.rag import RagQuery

from gateway_llms.app.utils.logs import LogApplication


router = APIRouter()


@router.post(
    path="/store",
    description="Transforme o seus documentos em vetores de indices"
)
async def documents_to_embeddings(
    request: Request,
    response: Response,
    file: UploadFile = File(...)
):
    log_user = LogApplication(request, request.body())

    await document_to_embeddings(file, log_user)

    return {"message": "Os vetores de indices forma salvos na aplicação"}


@router.post(
    path="/query",
    description="Transforme o seus documentos em vetores de indices"
)
async def response_query(
    request: Request,
    response: Response,
    rag_query: RagQuery
):
    log_user = LogApplication(request, await request.body())

    data = await storage_to_query(rag_query.text, rag_query.document_name, log_user)

    return data
