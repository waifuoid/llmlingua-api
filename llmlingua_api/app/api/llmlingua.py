from fastapi import APIRouter
from llmlingua_api.library.llmlingua import prompt_compressor
from llmlingua_api.app.models.request import CompressPromptRequest


router = APIRouter()

reranker = None


@router.get("/v1/compress_prompt")
async def compress_prompt(request: CompressPromptRequest):
    result = prompt_compressor.compress_prompt(**request.dict())

    response = {
        "data": result
    }

    return response
