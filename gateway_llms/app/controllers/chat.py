from gateway_llms.app.interfaces.chat import ChatLLMCompletion
from gateway_llms.app.utils.logs import LogApplication


async def get_chat_completion(
    chat_llm_completion: ChatLLMCompletion,
    log_user: LogApplication
):
    return "Ok"
