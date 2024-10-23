import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from loguru import logger
from endpoints.health import router as health_router
from endpoints.training import router as training_router

load_dotenv(os.getenv("ENV_FILE", ".env"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")

    yield

    logger.info("Shutting down...")


def factory() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(training_router)

    return app