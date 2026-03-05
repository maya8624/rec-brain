from fastapi import APIRouter

router = APIRouter(prefix="/api/ai", tags=["AI Chat"])


@router.get("/health")
async def health_check():
    """
        Returns a simple JSON response to prove the server is running.
    """
    return {"status": "healthy"}
