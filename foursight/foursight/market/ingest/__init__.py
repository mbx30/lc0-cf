"""Platform-specific ingest adapters (Lightweight + Medium tiers only)."""

from foursight.market.ingest.kalshi import load_kalshi_csv
from foursight.market.ingest.polymarket import load_polymarket_csv

__all__ = ["load_kalshi_csv", "load_polymarket_csv"]
