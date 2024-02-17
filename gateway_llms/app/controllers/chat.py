from gateway_llms.app.interfaces.chat import ChatLLMCompletion, ChatMessages
from gateway_llms.app.modules.chat.similarity import (
    get_similarity_messages_historical
)
from gateway_llms.app.modules.models.api_gemini import gemini_chat_completion
from gateway_llms.app.utils.logs import LogApplication, log_function


@log_function
async def chat_completion(
    chat_llm_completion: ChatLLMCompletion,
    log_user: LogApplication
):
    messages_similarity = []

    if chat_llm_completion.messages:
        messages_similarity = chat_llm_completion.messages

        for index in range(len(messages_similarity)):
            messages_similarity[index] = messages_similarity[index].dict()
    else:
        chat_llm_completion.messages = []

    if len(messages_similarity) > 0:
        messages_similarity = await get_similarity_messages_historical(
            chat_llm_completion,
            log_user
        )

    if chat_llm_completion.parameters:
        for key in chat_llm_completion.parameters:
            chat_llm_completion.system = chat_llm_completion.system.replace(
                "{{" + key + "}}", chat_llm_completion.parameters.get(key))
            chat_llm_completion.text = chat_llm_completion.text.replace(
                "{{" + key + "}}", chat_llm_completion.parameters.get(key))

    content = await gemini_chat_completion(
        chat_llm_completion.text,
        chat_llm_completion.system,
        messages_similarity,
        chat_llm_completion.generation_config.dict(),
        chat_llm_completion.safety_settings.dict(),
        log_user
    )

    chat_llm_completion.messages.append(ChatMessages(
        role="user",
        content=chat_llm_completion.text
    ))

    chat_llm_completion.messages.append(ChatMessages(
        role="assistant",
        content=content
    ))

    return chat_llm_completion.messages
