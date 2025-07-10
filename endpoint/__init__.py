from . import ragroutes
from fastapi import APIRouter

router = APIRouter()

router.include_router(ragroutes.router)


