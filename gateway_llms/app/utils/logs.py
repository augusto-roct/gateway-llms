import uuid
from fastapi import Request


class LogApplication:
    def __init__(self, request: Request, body: bytes):
        self.ip = request.client.host
        cookies = request.cookies
        headers = request.headers
        self.method = request.method
        self.path_params = request.path_params
        query_params = request.query_params
        self.id = uuid.uuid4()

        self.message = f"ID: {self.id}, IP: {self.ip}, Method: {self.method}, "
        f"path: {self.path_params}"

        print(self.message + f" Headers: {headers}")
        print(self.message + f" Cookies: {cookies}")
        print(self.message + f" Body: {body}")
        print(self.message + f" Query Params: {query_params}")
