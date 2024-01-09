import numpy as np
from gateway_llms.app.modules.models.api_gemini import gemini_embeddings
from gateway_llms.app.utils.logs import LogApplication, log_function


@log_function
async def get_embeddings(messages: list | str, log_user: LogApplication):
    list_embeddings = []

    for message in messages:
        embedding = await gemini_embeddings(message, log_user)
        list_embeddings.append(embedding)

    return np.array(list_embeddings)
