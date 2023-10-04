import uuid
from fastapi import Request


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
