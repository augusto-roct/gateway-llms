from fastapi import APIRouter, Request, Response

from gateway_llms.app.utils.logs import LogApplication


router = APIRouter()


@router.get(path="/")
async def root(
    request: Request,
    response: Response
):
    LogApplication(request, await request.body())
    return "API Gateway-llms is alive"
