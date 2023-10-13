import numpy as np
from gateway_llms.app.utils.logs import LogApplication, log_function


@log_function
async def get_embeddings(messages: list | str, log_user: LogApplication):
    return np.array([])
