#  ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}