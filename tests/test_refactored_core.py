import pytest
import pandas as pd
from datetime import datetime, timezone
from src.core.interfaces import MarketDataRequest
from src.data.duckdb_store import DataStore
from src.data.repository import DataRepository
from src.connectors.market.yfinance import YFinanceConnector

def test_datastore_initialization(tmp_path):
    pytest.importorskip("duckdb")
    db_path = tmp_path / "test.duckdb"
    store = DataStore(str(db_path))
    assert db_path.exists()

def test_market_data_request():
    req = MarketDataRequest(symbols=["AAPL"], timeframe="1d")
    assert req.symbols == ["AAPL"]
    assert req.timeframe == "1d"
    assert req.category == "market_data"

def test_yfinance_connector_caching(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    db_path = tmp_path / "test.duckdb"
    store = DataStore(str(db_path))
    
    # Mock yfinance to avoid actual network calls
    mock_df = pd.DataFrame({
        "Open": [150.0],
        "High": [155.0],
        "Low": [149.0],
        "Close": [152.0],
        "Volume": [1000000]
    }, index=pd.DatetimeIndex([datetime.now(timezone.utc)], name="Date"))
    monkeypatch.setattr("yfinance.download", lambda *_args, **_kwargs: mock_df)
    
    connector = YFinanceConnector({"enabled": True}, store=store)
    req = MarketDataRequest(symbols=["AAPL"], timeframe="1d")
    
    # First fetch (cold)
    res = connector.fetch_bars(req)
    assert "AAPL" in res
    assert not res["AAPL"].empty
    
    # Verify it's in the store
    cached = store.get_market_bars("AAPL", "1d", "yfinance", datetime.now(timezone.utc).replace(year=2000), datetime.now(timezone.utc))
    assert not cached.empty
