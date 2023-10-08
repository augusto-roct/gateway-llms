from gateway_llms.app.interfaces.chat import ChatLLMCompletion, ChatMessages
from gateway_llms.app.modules.chat.similarity import (
    get_similarity_messages_historical
)
from gateway_llms.app.modules.models.api_openai import openai_chat_completion
from gateway_llms.app.utils.logs import LogApplication


async def chat_completion(
    chat_llm_completion: ChatLLMCompletion,
    log_user: LogApplication
):
    messages_similarity = await get_similarity_messages_historical(
        chat_llm_completion,
        log_user
    )

    messages_similarity.append({
        "role": "user",
        "content": chat_llm_completion.text
    })

    content = await openai_chat_completion(
        chat_llm_completion.model,
        messages_similarity,
        log_user
    )

    chat_llm_completion.messages.append(ChatMessages(
        role="assistant",
        content=content
    ))

    return chat_llm_completion.messages
