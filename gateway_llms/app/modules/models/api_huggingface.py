import os
from llama_index.llms import HuggingFaceInferenceAPI
from llama_index.embeddings import HuggingFaceInferenceAPIEmbedding

from gateway_llms.app.utils.logs import LogApplication, log_function


HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


@log_function
def get_chat_model(config: dict, log_user: LogApplication):
    temperature = 0.1 if not config.get(
        "temperature"
    ) else config.get(
        "temperature"
    )

    return HuggingFaceInferenceAPI(
        model_name=config.get("chat_model_name"),
        token=HF_TOKEN,
        parameters={
            "temperature": temperature
        }
    )


@log_function
def get_embbeding_model(config: dict, log_user: LogApplication):
    return HuggingFaceInferenceAPIEmbedding(
        model_name=config.get("embedding_model_name"),
        token=HF_TOKEN
    )
