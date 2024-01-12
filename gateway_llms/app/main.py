from fastapi import FastAPI

from gateway_llms.__init__ import __version__
from gateway_llms.app.routers import index, chat, embeddings, rag


app = FastAPI(
    version=__version__
)

app.include_router(index.router, prefix="")
app.include_router(chat.router, prefix="/chat")
app.include_router(embeddings.router, prefix="/embedding")
app.include_router(rag.router, prefix="/rag")
