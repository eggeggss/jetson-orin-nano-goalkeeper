from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from .routes import router
import os

_TMPL_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=_TMPL_DIR)


def create_app() -> FastAPI:
    app = FastAPI(title="備品室門禁系統", version="1.0.0", docs_url="/docs")
    app.include_router(router)
    return app
