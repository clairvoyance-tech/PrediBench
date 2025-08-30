from pydantic import BaseModel


class DataPoint(BaseModel):
    date: str
    value: float


class LeaderboardEntry(BaseModel):
    id: str
    model: str
    final_cumulative_pnl: float
    trades: int
    profit: int
    lastUpdated: str
    trend: str
    pnl_history: list[DataPoint]
    avg_brier_score: float


class Stats(BaseModel):
    topFinalCumulativePnl: float
    avgPnl: float
    totalTrades: int
    totalProfit: int