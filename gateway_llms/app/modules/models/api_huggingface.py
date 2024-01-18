import os
from llama_index import ServiceContext
from llama_index.llms import HuggingFaceInferenceAPI
from llama_index.embeddings import HuggingFaceInferenceAPIEmbedding

from gateway_llms.app.interfaces.chat import ChatConfig
from gateway_llms.app.interfaces.rag import RagConfig
from gateway_llms.app.utils.logs import LogApplication, log_function


HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


@log_function
def get_service_context(
    system_prompt: str | None,
    config: RagConfig | ChatConfig,
    log_user: LogApplication
):
    if isinstance(config, ChatConfig):
        generation_config = config.dict()
        service_config = {}
    else:
        generation_config = {"temperature": 0.1}
        service_config = config.dict()

    llm = HuggingFaceInferenceAPI(
        model_name=service_config.get("chat_model_name"),
        token=HF_TOKEN,
        parameters={
            "temperature": generation_config.get("temperature")
        }
    )

    embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name=service_config.get("embedding_model_name"),
        token=HF_TOKEN
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt=system_prompt,
        chunk_size=service_config.get("chunk_size"),
        chunk_overlap=service_config.get("chunk_overlap")
    )

    return service_context
