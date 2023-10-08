from gateway_llms.app.interfaces.chat import ChatLLMCompletion
from gateway_llms.app.modules.chat.messages import get_similarity_messages_historical
from gateway_llms.app.utils.logs import LogApplication


async def get_chat_completion(
    chat_llm_completion: ChatLLMCompletion,
    log_user: LogApplication
):
    messages_similarity = await get_similarity_messages_historical(
        chat_llm_completion.text,
        chat_llm_completion.messages,
        log_user
    )
    return "Ok"
