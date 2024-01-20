import os
from starlette.concurrency import run_in_threadpool
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from llama_index.embeddings import GooglePaLMEmbedding

from gateway_llms.app.utils.logs import LogApplication, log_function


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)


@log_function
async def transform_history(history: list, log_user: LogApplication):
    for message in history:
        message.update({"parts": [{"text": message.get("content")}]})
        del message["content"]

        if message.get("role") == "assistant":
            message.update({"role": "model"})

    return history


@log_function
async def gemini_chat_completion(
    message: str,
    system: str,
    history: list,
    generation_config: dict,
    safety_settings: dict,
    log_user: LogApplication
):
    model = genai.GenerativeModel('gemini-pro')

    history.insert(0, {"role": "user", "content": "Quem é você?"})
    history.insert(1, {"role": "assistant", "content": system})

    chat_history = await transform_history(history, log_user)

    chat = model.start_chat(history=chat_history)

    response = await chat.send_message_async(
        message,
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    return response.text


@log_function
async def gemini_embeddings(
    text: str,
    log_user: LogApplication
):
    response = await run_in_threadpool(
        genai.embed_content,
        'models/embedding-001',
        text
    )

    return response.get("embedding")


@log_function
def get_embbeding_model(log_user: LogApplication):
    model_name = "models/embedding-gecko-001"

    return GooglePaLMEmbedding(
        model_name=model_name,
        api_key=GOOGLE_API_KEY,
    )


@log_function
def get_chat_model(generation_config: dict, safety_settings: dict, log_user: LogApplication):
    temperature = 0.1 if not generation_config.get(
        "temperature"
    ) else generation_config.get(
        "temperature"
    )

    return Gemini(
        GOOGLE_API_KEY,
        temperature=temperature,
        max_tokens=None,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
