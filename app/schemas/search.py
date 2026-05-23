from datetime import date, datetime

from pydantic import BaseModel, field_validator

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


class TenancyDocsRequest(BaseModel):
    property_id: str
    tenant_id: str


class TenancyDetails(BaseModel):
    agreement_type: str
    commencement: date
    end_date: date
    rent_amount: float
    rent_frequency: str
    rent_due_day: str
    payment_method: str
    payment_bsb: str | None = None
    payment_account: str | None = None
    bond_amount: float
    bond_receipt_no: str

    @field_validator("commencement", "end_date", mode="before")
    @classmethod
    def parse_date(cls, v: str) -> date:
        if isinstance(v, date):
            return v
        for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(v, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unrecognised date format: {v}")


class TenancyDocsResponse(BaseModel):
    tenancy: TenancyDetails | None = None
