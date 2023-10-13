import time
import uuid
from fastapi import Request


def log_function(func):
    async def async_functions(*args, **kwargs):
        start = time.time()

        log_application = args[-1]
        if isinstance(log_application, LogApplication):
            log_application.log_info("INFO", f"Call {func.__name__}")

        result = await func(*args, **kwargs)

        if isinstance(log_application, LogApplication):
            finish = time.time() - start
            log_application.log_info(
                "INFO",
                f"Return {func.__name__}, time execution: {finish: .3f} seconds"
            )

        return result

    return async_functions


class LogApplication:
    def __init__(self, request: Request, body: bytes):
        self.ip = request.client.host
        cookies = request.cookies
        headers = request.headers
        self.method = request.method
        self.path = request.scope.get("path")
        query_params = request.query_params
        self.id = uuid.uuid4()

        self.message = f"ID: {self.id}, IP: {self.ip}, " \
            f"Method: {self.method}, path: {self.path}, "

        print(self.message + f"Headers: {headers}")
        print(self.message + f"Cookies: {cookies}")
        print(self.message + f"Body: {body}")
        print(self.message + f"Query Params: {query_params}")

    def log_info(self, level: str, message: str):

        print(self.message + f"{level}, Status: {message}")
