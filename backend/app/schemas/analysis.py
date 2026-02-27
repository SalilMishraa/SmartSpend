from pydantic import BaseModel
from typing import Dict, List


class DailySpending(BaseModel):
    date: str
    amount: float


class TopDay(BaseModel):
    date: str
    amount: float


class Anomaly(BaseModel):
    date: str
    category: str
    amount: float
    reason: str


class Metrics(BaseModel):
    total_spent: float
    category_spending: Dict[str, float]
    daily_spending: List[DailySpending]
    avg_daily_spending: float
    top_3_days: List[TopDay]
    anomalies: List[Anomaly]


class AnalyzeRequest(BaseModel):
    raw_data: str
    spending_limit: float


class AnalyzeResponse(BaseModel):
    metrics: Metrics
    ai_suggestions: str
    dropped_rows: int