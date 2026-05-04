"""
HTTP client for the .NET backend deposit API.
"""
import logging
from app.core.constants import InternalRoutes
from app.core.exceptions import BackendClientError, DepositServiceError
from app.schemas.deposit import DepositResult
from app.services.backend_client import BackendClient

logger = logging.getLogger(__name__)


class DepositService:
    """HTTP client for deposit operations against the .NET backend."""

    def __init__(self, client: BackendClient):
        self._client = client

    async def get_my_deposit(self, listing_id: str, user_id: str) -> DepositResult:
        """Fetch the holding deposit for a user on a specific listing."""
        try:
            data = await self._client.get(InternalRoutes.my_deposit(listing_id, user_id))
        except BackendClientError as exc:
            raise DepositServiceError(
                f"Failed to fetch deposit: {exc}", exc.status_code) from exc

        return DepositResult(success=True, **data)
