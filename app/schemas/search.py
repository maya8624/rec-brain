from pydantic import BaseModel

from app.schemas.property import Listing


class TenantPreference(BaseModel):
    suburbs: list[str]
    maxRent: float | None = None
    minBeds: int | None = None
    maxBeds: int | None = None
    petFriendly: bool = False
    availableWithinDays: int | None = None


class PreferenceSearchResponse(BaseModel):
    message: str
    listings: list[Listing]
    display_count: int
    total_count: int
    has_more: bool


class SuburbSummaryRequest(BaseModel):
    suburbs: list[str]


class SuburbRents(BaseModel):
    one_bedroom: str | None = None
    two_bedroom: str | None = None
    three_bedroom: str | None = None


class SuburbProfile(BaseModel):
    name: str
    description: str
    rents: SuburbRents
    vacancy_rate: str | None = None
    trend: str | None = None


class SuburbSummaryResponse(BaseModel):
    suburbs: list[SuburbProfile] = []
