#  ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

from fastapi import APIRouter, Form

from service.rag.rag_pipeline import RAGPipeline

router = APIRouter()

rag_pipeline = RAGPipeline()

@router.post("/chat")
async def chat(
    query: str = Form(...),
    collection: str = Form(...),
    tenant_id: str = Form(...),
    top_k: int = Form(5)
):
    result = await rag_pipeline.query(query, collection, tenant_id, top_k)
    return result
