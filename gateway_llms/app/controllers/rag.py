import zipfile
from fastapi import UploadFile
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from gateway_llms.app.interfaces.rag import RagConfig, RagQuery

from gateway_llms.app.utils.logs import LogApplication, log_function
from gateway_llms.app.modules.models.api_gemini import get_service_context


PATH_DATA_DOCUMENTS = "gateway_llms/app/data/documents/"
PATH_DATA_EMBEDDINGS = "gateway_llms/app/data/embeddings/"


@log_function
def save_file(file: UploadFile, log_user: LogApplication):
    file_name = file.filename.replace(".zip", "")

    with zipfile.ZipFile(file.file, 'r') as zip_ref:
        zip_ref.extractall(
            f"{PATH_DATA_DOCUMENTS}{file_name}"
        )

    del file


@log_function
async def document_to_embeddings(file_name: str, rag_config: RagConfig, log_user: LogApplication):
    service_context = get_service_context("", rag_config, log_user)

    documents = SimpleDirectoryReader(
        f"{PATH_DATA_DOCUMENTS}{file_name}"
    ).load_data()

    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context
    )

    index.storage_context.persist(
        f"{PATH_DATA_EMBEDDINGS}{file_name}"
    )


@log_function
async def storage_to_query(rag_query: RagQuery, log_user: LogApplication):
    service_context = get_service_context(
        rag_query.system_prompt,
        rag_query.config,
        log_user
    )

    storage_context = StorageContext.from_defaults(
        persist_dir=f"{PATH_DATA_EMBEDDINGS}{rag_query.document_name}"
    )

    index = load_index_from_storage(
        storage_context,
        service_context=service_context
    )

    query_engine = index.as_query_engine()
    response = query_engine.query(rag_query.text)

    return {"message": response.response}
