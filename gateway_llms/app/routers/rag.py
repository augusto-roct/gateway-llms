from fastapi import APIRouter, Request, Response, File, UploadFile
from gateway_llms.app.controllers.rag import document_to_embeddings, save_file, storage_to_query
from gateway_llms.app.interfaces.rag import RagDocument, RagQuery

from gateway_llms.app.utils.logs import LogApplication


router = APIRouter()


@router.post(
    path="/upload",
    description="Salve os documentos dentro da aplicação"
)
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
):
    log_user = LogApplication(request, request.body())

    save_file(file, log_user)

    return {"message": "O documento foi salvo na aplicação"}


@router.post(
    path="/store",
    description="Transforme o seus documentos em vetores de indices"
)
async def documents_to_embeddings(
    request: Request,
    response: Response,
    rag_document: RagDocument
):
    log_user = LogApplication(request, await request.body())

    try:
        await document_to_embeddings(rag_document.name, rag_document.config, log_user)

        return {"message": "Os vetores de indices forma salvos na aplicação"}
    except Exception:
        response.status_code = 500
        return {"message": "Ocorreu um erro durante a execução"}


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

    data = await storage_to_query(rag_query, log_user)

    return data
