import os
import google.generativeai as genai
from starlette.concurrency import run_in_threadpool

from gateway_llms.app.utils.logs import LogApplication, log_function


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


@log_function
async def transform_history(history: list):
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
    config: dict,
    log_user: LogApplication
):
    model = genai.GenerativeModel('gemini-pro')

    history.insert(0, {"role": "user", "content": "Quem é você?"})
    history.insert(1, {"role": "assistant", "content": system})

    chat_history = await transform_history(history)

    chat = model.start_chat(history=chat_history)

    response = await chat.send_message_async(
        message,
        generation_config=config
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
