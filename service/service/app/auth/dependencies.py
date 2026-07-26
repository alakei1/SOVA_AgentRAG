import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from service.utils.config import settings

security = HTTPBearer()

async def get_current_tenant(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Извлекает tenant_id из JWT-токена.
    В токене должно быть поле "tenant_id".
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        tenant_id = payload.get("tenant_id")
        if not tenant_id:
            raise HTTPException(status_code=401, detail="Missing tenant_id in token")
        return tenant_id
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")  # noqa: B904
