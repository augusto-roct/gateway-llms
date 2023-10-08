import os
from typing import List
from fastapi.concurrency import run_in_threadpool
import openai

from gateway_llms.app.utils.logs import LogApplication


openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")


async def call_openai_api(func, args: dict):

    response = await run_in_threadpool(
        func,
        *args
    )

    return response


async def openai_chat_completion(
    model: str,
    messages: List[dict],
    log_user: LogApplication
):
    func = openai.ChatCompletion.create

    args = {
        "model": model,
        "messages": messages
    }

    response = await call_openai_api(func, args)

    return response.choices[0].message.content
