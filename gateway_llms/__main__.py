import uvicorn

from gateway_llms.app.main import app


if __name__ == "__main__":
    print("Started Server")

    uvicorn.run(app, host="0.0.0.0", port="8000", reload=True)
