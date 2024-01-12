import zipfile
from fastapi import UploadFile
from llama_index import VectorStoreIndex, SimpleDirectoryReader

from gateway_llms.app.utils.logs import LogApplication, log_function
from gateway_llms.app.modules.models.api_gemini import service_context


@log_function
async def document_to_embeddings(file: UploadFile, log_user: LogApplication):
    file_name = file.filename.replace(".zip", "")

    with zipfile.ZipFile(file.file, 'r') as zip_ref:
        zip_ref.extractall(
            f"gateway_llms/app/data/documents/{file_name}"
        )

    del file

    documents = SimpleDirectoryReader(
        f"gateway_llms/app/data/documents/{file_name}").load_data()

    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context)

    index.storage_context.persist(
        f"gateway_llms/app/data/embeddings/{file_name}")
