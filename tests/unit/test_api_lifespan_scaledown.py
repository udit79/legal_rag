import asyncio

from fastapi import FastAPI

import src.api.main as api_main
from src.ingestion.scaledown_client import ScaleDownClient


class StubProvider:
    def __init__(self) -> None:
        self.flushed = False
        self.stopped = False

    def force_flush(self, timeout_millis: int) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.stopped = True


class StubDatabase:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_application_lifespan_shares_and_closes_scaledown_http_client(
    monkeypatch,
) -> None:
    provider = StubProvider()
    database = StubDatabase()
    dotenv_loads: list[bool] = []
    monkeypatch.setattr(api_main, "load_dotenv", lambda: dotenv_loads.append(True))
    monkeypatch.setattr(api_main, "configure_telemetry", lambda: (provider, None))
    monkeypatch.setattr(api_main, "create_db_connect", lambda: database)
    monkeypatch.setattr(api_main.FastAPIInstrumentor, "instrument_app", lambda app: None)
    monkeypatch.setattr(
        api_main.FastAPIInstrumentor, "uninstrument_app", lambda app: None
    )

    async def run_lifespan() -> None:
        app = FastAPI()
        manager = api_main.lifespan(app)
        await manager.__aenter__()
        http_client = app.state.scaledown_http_client
        assert not http_client.is_closed
        assert isinstance(app.state.scaledown_client, ScaleDownClient)
        assert app.state.scaledown_client._http_client is http_client
        await manager.__aexit__(None, None, None)
        assert http_client.is_closed

    asyncio.run(run_lifespan())

    assert database.closed
    assert dotenv_loads == [True]
    assert provider.flushed
    assert provider.stopped
