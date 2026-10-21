lazy import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError,ServiceUnavailable,DriverError
load_dotenv()
url=os.getenv("NEO4J_URI")
username=os.getenv("NEO4J_USERNAME")
password=os.getenv("NEO4J_PASSWORD")
database=os.getenv("NEO4J_DATABASE")
class DbConnect:
    def __init__(self,*,url,username,password,database):
        self.url=url
        self.username=username
        self.password=password
        self.database=database
        if self.url is None or self.username is None or self.password is None or self.database is None:
            raise ValueError("Set NEO4J_URI,NEO4J_USERNAME,NEO4J_PASSWORD in .env")
        self.driver=GraphDatabase.driver(self.url,auth=(self.username,self.password))
    def verify_connection(self):
        try:
            self.driver.verify_connectivity()
        except ServiceUnavailable as exc:
            raise ConnectionError(
                "Could not connect to Neo4j. Check that the database is running "
                "and NEO4J_URI is correct."
            ) from exc
        except AuthError as exc:
            raise ConnectionError(
                "Neo4j authentication failed. Check your username and password."
            ) from exc
        except DriverError as exc:
            raise ConnectionError(
                "Could not connect to driver."
            )from exc
    def close(self):
        self.driver.close