from fastapi import APIRouter, Request, Response
from gateway_llms.app.interfaces.chat import ChatLLMCompletion
from gateway_llms.app.controllers.chat import get_chat_completion

from gateway_llms.app.utils.logs import LogApplication


router = APIRouter()


@router.post(
    path="/completion",
    description="Converse com os modelos LLMs com histórico de conversa"
)
async def chat_completion(
    request: Request,
    response: Response,
    chat_llm_completion: ChatLLMCompletion
):
    log_user = LogApplication(request, await request.body())

    data = await get_chat_completion(chat_llm_completion, log_user)
    return data
