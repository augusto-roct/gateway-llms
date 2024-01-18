from gateway_llms.app.interfaces.chat import ChatConfig
from gateway_llms.app.modules.models import api_gemini, api_huggingface
from gateway_llms.app.utils.logs import LogApplication, log_function
from gateway_llms.app.interfaces.rag import RagConfig, RagQuery
import zipfile
from fastapi import UploadFile
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
import nest_asyncio
nest_asyncio.apply()


PATH_DATA_DOCUMENTS = "gateway_llms/app/data/documents/"
PATH_DATA_EMBEDDINGS = "gateway_llms/app/data/embeddings/"


@log_function
def get_service_context(
    system_prompt: str | None,
    config: RagConfig | ChatConfig,
    log_user: LogApplication
):
    config = config.dict()
    if "gemini" in config.get("chat_model_name").lower():
        llm = api_gemini.get_chat_model(
            config,
            log_user
        )
    else:
        llm = api_huggingface.get_chat_model(
            config,
            log_user
        )

    if "gemini" in config.get("embedding_model_name").lower():
        embed_model = api_gemini.get_embbeding_model(
            log_user
        )
    else:
        embed_model = api_huggingface.get_embbeding_model(
            config,
            log_user
        )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt=system_prompt,
        chunk_size=config.get("chunk_size"),
        chunk_overlap=config.get("chunk_overlap")
    )

    return service_context


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
    service_context = get_service_context(
        "",
        rag_config,
        log_user
    )

    documents = SimpleDirectoryReader(
        f"{PATH_DATA_DOCUMENTS}{file_name}"
    ).load_data()

    try:
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
    except Exception:
        raise ValueError()

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
