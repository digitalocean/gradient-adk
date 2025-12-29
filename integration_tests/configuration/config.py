import os
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # Fields expected from .env
    BASE_URL: str = Field(..., alias="BASE_URL")
    API_TOKEN: str = Field(..., alias="API_TOKEN")

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        extra="forbid",  # Optional: ensure no undeclared keys sneak in
        populate_by_name=True,
    )
