import os
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable, DriverError
from opentelemetry import trace
from opentelemetry.trace import Status,StatusCode
load_dotenv()
tracer=trace.get_tracer("graph_rag.neo4j")
class DbConnect:
    def __init__(self, *, url, username, password, database):
        config = {
            "NEO4J_URI": url,
            "NEO4J_USERNAME": username,
            "NEO4J_PASSWORD": password,
            "NEO4J_DATABASE": database,
        }
        missing = [
            name for name, value in config.items()
            if not isinstance(value, str) or not value.strip()
        ]
        if missing:
            raise ValueError(
                f"Missing or invalid configuration: {', '.join(missing)}"
            )

        self.url = url
        self.username = username
        self.database = database

        self.driver = AsyncGraphDatabase.driver(
            self.url,
            auth=(self.username, password),
        )
    async def verify_connection(self):
        with tracer.start_as_current_span("neo4j.verify_connection") as span:
            try:
                await self.driver.verify_connectivity()
                span.set_attribute("db.system", "neo4j")
            except AuthError as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, description=str(exc)))
                raise ConnectionError(
                    "Neo4j authentication failed. Check the credentials."
                ) from exc
            except ServiceUnavailable as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, description=str(exc)))
                raise ConnectionError(
                    "Neo4j is unavailable. Check that the database is running "
                    "and the URI is correct."
                ) from exc
            except DriverError as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, description=str(exc)))
                raise ConnectionError(
                    "Neo4j driver error during connection verification."
                ) from exc

    async def close(self):
        await self.driver.close()
def create_db_connect():
    return DbConnect(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )