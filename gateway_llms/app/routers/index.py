from fastapi import APIRouter


router = APIRouter()


@router.get(path="/")
async def root():
    return "API Gateway-llms is alive"
