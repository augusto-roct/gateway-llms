from fastapi import FastAPI

from gateway_llms.__init__ import __version__
from gateway_llms.app.routers import index


app = FastAPI(
    version=__version__
)

# app.add_route(path="/", route=index.router)
app.include_router(index.router, prefix="")
