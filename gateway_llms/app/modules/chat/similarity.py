import numpy as np

from gateway_llms.app.interfaces.chat import ChatLLMCompletion
from gateway_llms.app.modules.chat.embeddings import get_embeddings
from gateway_llms.app.utils.logs import LogApplication, log_function


@log_function
async def get_similarity_messages_historical(
    chat_llm_completion: ChatLLMCompletion,
    log_user: LogApplication
):
    list_index_messages = []
    messages_assistant = []

    index_system = 0

    for index, message in enumerate(chat_llm_completion.messages):
        if message.role == "system":
            index_system = index

        if message.role == "assistant":
            list_index_messages.append(index)
            messages_assistant.append(message.content)

    embeddings_user = await get_embeddings(chat_llm_completion.text, log_user)
    embeddings_assistant = await get_embeddings(messages_assistant, log_user)

    del messages_assistant

    similarity = np.dot(embeddings_assistant, embeddings_user)
    similarity = np.argsort(similarity)[:3]

    del embeddings_assistant
    del embeddings_user

    messages_similarity = []

    if chat_llm_completion.messages[index_system].role == "system":
        messages_similarity.append({
            "role": "system",
            "content": chat_llm_completion.messages[index_system].content
        })

    for index in similarity:
        messages_similarity.append({
            "role": "user",
            "content": chat_llm_completion.messages[index-1].content
        })
        messages_similarity.append({
            "role": "assistant",
            "content": chat_llm_completion.messages[index].content
        })

    return messages_similarity
