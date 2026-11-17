"""Response contracts of the API (Pydantic)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["licit", "illicit"]


class Health(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]


class Prediction(BaseModel):
    run_id: str
    tx_id: int
    time_step: int = Field(ge=1, le=49)
    score: float = Field(ge=0, le=1)
    predicted: Verdict
    label: Verdict | None


class Alerts(BaseModel):
    run_id: str
    time_step: int = Field(ge=1, le=49)
    alerts: list[Prediction]


class StepMetrics(BaseModel):
    time_step: int
    n_illicit: int
    precision: float
    recall: float
    f1: float


class Metrics(BaseModel):
    run_id: str
    steps: list[StepMetrics]


class Run(BaseModel):
    run_id: str
    name: str
    model: str
    seed: int
    train_seconds: float
    f1_all: float
    f1_35_42: float
    f1_43_49: float


class StreamAlert(BaseModel):
    run_id: str
    tx_id: int
    time_step: int = Field(ge=1, le=49)
    score: float = Field(ge=0, le=1)
    scored_at: datetime


class StreamAlerts(BaseModel):
    run_id: str
    time_step: int
    alerts: list[StreamAlert]
