from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class SourceType(str, Enum):
    CRM = "CRM"
    WEB = "WEB"
    EMAIL = "EMAIL"
    SOCIAL_MEDIA = "SOCIAL_MEDIA"
    OTHER = "OTHER"


class Search(BaseModel):
    query: str
    source_type: SourceType
    results_limit: Optional[int] = Field(10)
    is_priority: Optional[bool] = None
    tags: Optional[list[str]] = None


class MultiSearch(BaseModel):
    queries: list[Search]
    user_id: Optional[str]
