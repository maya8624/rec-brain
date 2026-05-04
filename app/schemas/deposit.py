from pydantic import BaseModel


class DepositResult(BaseModel):
    success: bool
    id: str = ""
    user_id: str = ""
    property_id: str = ""
    listing_id: str = ""
    amount: float = 0.0
    ispaid: bool
    currency: str | None = None
    stripe_session_id: str | None = None
    status: str | None = None
    paid_at_utc: str | None = None
    session_url: str | None = None
    error: str | None = None
