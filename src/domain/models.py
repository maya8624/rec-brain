from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship, declarative_base
from pydantic import BaseModel, Field

Base = declarative_base()

# --- SQL Database Models (SQLAlchemy) ---


class Agent(Base):
    __tablename__ = 'agents'
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    phone = Column(String(50))
    listings = relationship("Listing", back_populates="agent")


class Listing(Base):
    __tablename__ = 'listings'
    id = Column(Integer, primary_key=True)
    address = Column(String(500), nullable=False)
    city = Column(String(200), nullable=False)
    price = Column(Float, nullable=False)
    bedrooms = Column(Integer)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    agent_id = Column(Integer, ForeignKey('agents.id'))
    agent = relationship("Agent", back_populates="listings")

# --- API Data Models (Pydantic DTOs) ---
# These define what the .NET backend is allowed to send/receive


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    question: str = Field(..., min_length=1, max_length=300)


class ChatResponse(BaseModel):
    answer: str
    source: str  # "vector_db" or "sql_db"
