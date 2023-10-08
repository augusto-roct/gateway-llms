from typing import List
import numpy as np

from gateway_llms.app.interfaces.chat import ChatMessages
from gateway_llms.app.modules.embeddings.transformer import get_embeddings
from gateway_llms.app.utils.logs import LogApplication


async def get_similarity_messages_historical(
    text: str,
    messages: List[ChatMessages],
    log_user: LogApplication
):
    list_index_messages = []
    messages_assistant = []

    index_system = 0

    for index, message in enumerate(messages):
        if message.role == "system":
            index_system = index

        if message.role == "assistant":
            list_index_messages.append(index)
            messages_assistant.append(message.content)

    embeddings_user = await get_embeddings(text, log_user)
    embeddings_assistant = await get_embeddings(messages_assistant, log_user)

    del messages_assistant

    similarity = np.dot(embeddings_assistant, embeddings_user)
    similarity = np.argsort(similarity)[:3]

    del embeddings_assistant
    del embeddings_user

    messages_similarity = []

    if messages[index_system].role == "system":
        messages_similarity.append(ChatMessages(
            role="system",
            content=messages[index_system].content
        ))

    for index in similarity:
        messages_similarity.append(ChatMessages(
            role="user",
            content=messages[index-1].content
        ))
        messages_similarity.append(ChatMessages(
            role="assistant",
            content=messages[index].content
        ))

    return messages_similarity
