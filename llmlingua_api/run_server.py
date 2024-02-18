from llmlingua_api.library.restapi import app as fastapi_app
from llmlingua_api.library.restapi import load_router_callback
from llmlingua_api.library.autoload import LoadFile
from llmlingua_api.library.config import config


import os
import uvicorn


def run_server():
    load_file = LoadFile(callback=load_router_callback)
    load_file.root_path = os.path.dirname(os.path.abspath(__file__))
    load_file.loop_up("app/api")
    uvicorn.run(fastapi_app, host=config.SERVER_HOST, port=config.SERVER_PORT)
