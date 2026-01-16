from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MongoIntegration:
    available: bool
    client: Any | None
    db: Any | None
    generations: Any | None


def init_mongo(mongodb_uri: str | None) -> MongoIntegration:
    if not mongodb_uri:
        return MongoIntegration(available=False, client=None, db=None, generations=None)
    try:
        from pymongo.mongo_client import MongoClient
        from pymongo.server_api import ServerApi

        client = MongoClient(mongodb_uri, server_api=ServerApi("1"))
        client.admin.command("ping")
        db = client["PyReflect"]
        generations = db["generations"]
        print("MongoDB connected: PyReflect")
        return MongoIntegration(available=True, client=client, db=db, generations=generations)
    except Exception as exc:
        print(f"Warning: MongoDB connection failed: {exc}")
        return MongoIntegration(available=False, client=None, db=None, generations=None)


async def mongo_keepalive(client: Any) -> None:
    import asyncio

    while True:
        try:
            await asyncio.sleep(300)
            client.admin.command("ping")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            print(f"MongoDB keepalive ping failed: {exc}")

