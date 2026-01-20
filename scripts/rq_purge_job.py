#!/usr/bin/env python3
"""Purge a stuck RQ job from Redis.

This is meant for the "zombie job" case where the UI/API shows a job as started
forever, but the worker that was running it is gone.

It performs best-effort cleanup:
- Removes the job from the queue list (rq:queue:<queue>)
- Removes the job from registries (rq:wip:<queue>, rq:started:<queue>, rq:finished:<queue>, rq:failed:<queue>)
- Deletes job keys (rq:job:<id>, rq:results:<id>, and a few common companion keys)
- Optionally removes a stale worker record (rq:workers + rq:worker:<name>)

By default it runs in dry-run mode. Pass --execute to actually modify Redis.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


def _normalize_redis_url(value: str) -> str:
    value = (value or "").strip()
    if value.lower().startswith("redis_url="):
        value = value.split("=", 1)[1].strip()
    if value[:1] in {'"', "'"}:
        value = value[1:]
    if value[-1:] in {'"', "'"}:
        value = value[:-1]
    value = value.strip()

    if not value:
        return ""
    if value.startswith(("redis://", "rediss://", "unix://")):
        return value
    if value.startswith("//"):
        return "redis:" + value
    if value.startswith(("tcp://", "redis+tcp://")):
        return "redis://" + value.split("://", 1)[1]
    if value.startswith(("ssl://", "tls://", "redis+ssl://")):
        return "rediss://" + value.split("://", 1)[1]
    if "://" not in value:
        return f"redis://{value}"
    return value


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso8601(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        # Accept the common "...Z" suffix.
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@dataclass(frozen=True)
class Plan:
    queue_key: str
    registry_zset_keys: tuple[str, ...]
    job_keys: tuple[str, ...]
    worker_set_key: str
    worker_key_prefix: str


def _build_plan(queue_name: str, job_id: str) -> Plan:
    # RQ uses list for queue and sorted sets for registries.
    queue_key = f"rq:queue:{queue_name}"

    # Different RQ versions / configurations can use different started registry keys.
    # In this repo we've observed `rq:wip:<queue>`.
    registry_zset_keys = (
        f"rq:wip:{queue_name}",
        f"rq:started:{queue_name}",
        f"rq:finished:{queue_name}",
        f"rq:failed:{queue_name}",
    )

    job_keys = (
        f"rq:job:{job_id}",
        f"rq:results:{job_id}",
        # Common companion keys (best-effort; may not exist).
        f"rq:job:{job_id}:dependents",
        f"rq:job:{job_id}:dependencies",
    )

    return Plan(
        queue_key=queue_key,
        registry_zset_keys=registry_zset_keys,
        job_keys=job_keys,
        worker_set_key="rq:workers",
        worker_key_prefix="rq:worker:",
    )


def _iter_zset_members(redis_conn, key: str) -> list[str]:
    try:
        # ZRANGE 0 -1
        raw = redis_conn.zrange(key, 0, -1)
        return [m.decode("utf-8", errors="replace") if isinstance(m, (bytes, bytearray)) else str(m) for m in raw]
    except Exception:
        return []


def _safe_decode(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _find_started_registry_members(members: Iterable[str], job_id: str) -> list[str]:
    hits: list[str] = []
    for m in members:
        if m == job_id or m.startswith(job_id + ":"):
            hits.append(m)
    return hits


def _get_job_meta_updated_at(redis_conn, job_key: str) -> str | None:
    # Our app writes meta.updated_at into job.meta, but RQ stores meta pickled.
    # Fetching it reliably via Redis alone is non-trivial; attempt via rq.Job if available.
    try:
        from rq.job import Job

        job_id = job_key.split("rq:job:", 1)[1]
        job = Job.fetch(job_id, connection=redis_conn)
        meta = job.meta or {}
        updated_at = meta.get("updated_at")
        if updated_at is None:
            return None
        return str(updated_at)
    except Exception:
        return None


def _maybe_find_workers_for_job(redis_conn, plan: Plan, job_id: str) -> list[str]:
    # Best-effort: iterate rq:workers set and check each worker hash for a current job id.
    try:
        worker_names_raw = redis_conn.smembers(plan.worker_set_key)
    except Exception:
        return []

    worker_names = [_safe_decode(w) for w in worker_names_raw]
    matches: list[str] = []
    for name in worker_names:
        if not name:
            continue
        key = plan.worker_key_prefix + name
        try:
            h = redis_conn.hgetall(key)
        except Exception:
            continue
        # Typical keys: current_job_id, state, last_heartbeat, etc.
        values = {(_safe_decode(k)): (_safe_decode(v)) for k, v in (h or {}).items()}
        for candidate_field in ("current_job_id", "job_id", "current_job"):
            if values.get(candidate_field) == job_id:
                matches.append(name)
                break

    return matches


class _RedisCLI:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url

    def _run(self, *args: str) -> list[str]:
        # Use --raw to make parsing easy and stable.
        cmd = ["redis-cli", "-u", self.redis_url, "--raw", *args]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "redis-cli failed").strip())
        out = (proc.stdout or "").splitlines()
        return [line.rstrip("\n") for line in out]

    def ping(self) -> None:
        out = self._run("PING")
        if not out or out[0].strip().upper() != "PONG":
            raise RuntimeError("PING did not return PONG")

    def lrem(self, key: str, count: int, value: str) -> None:
        self._run("LREM", key, str(count), value)

    def zrange(self, key: str, start: int, stop: int) -> list[str]:
        return self._run("ZRANGE", key, str(start), str(stop))

    def zrem(self, key: str, member: str) -> None:
        self._run("ZREM", key, member)

    def delete(self, *keys: str) -> None:
        if not keys:
            return
        self._run("DEL", *keys)

    def smembers(self, key: str) -> list[str]:
        return self._run("SMEMBERS", key)

    def srem(self, key: str, member: str) -> None:
        self._run("SREM", key, member)

    def hgetall(self, key: str) -> dict[str, str]:
        out = self._run("HGETALL", key)
        # redis-cli returns alternating key/value lines.
        if not out:
            return {}
        if len(out) % 2 != 0:
            # Best-effort: ignore trailing line.
            out = out[:-1]
        it = iter(out)
        return {str(k): str(v) for k, v in zip(it, it)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Purge a stuck RQ job from Redis")
    parser.add_argument("--job-id", required=True, help="RQ job id to purge")
    parser.add_argument("--queue", default="training", help="RQ queue name (default: training)")
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", ""),
        help="Redis URL (default: env REDIS_URL)",
    )
    parser.add_argument(
        "--min-stale-s",
        type=int,
        default=600,
        help="Refuse purge unless meta.updated_at is older than this (default: 600). Use --force to bypass.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass staleness check (dangerous)",
    )
    parser.add_argument(
        "--worker",
        default=None,
        help="Optional worker name to remove (e.g. gpu-t4-modal-123). If omitted, will try to auto-detect.",
    )
    parser.add_argument(
        "--clean-workers",
        action="store_true",
        help="Remove worker records that claim they are running this job.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually apply changes (default is dry-run).",
    )

    args = parser.parse_args(argv)

    redis_url = _normalize_redis_url(args.redis_url)
    if not redis_url:
        print("ERROR: Redis URL missing. Set REDIS_URL or pass --redis-url.", file=sys.stderr)
        return 2

    # Prefer redis-py if available; fall back to redis-cli so this script can run
    # on minimal servers.
    redis_conn = None
    redis_cli = None
    redis_mode = "redis-py"
    try:
        from redis import Redis  # type: ignore

        redis_conn = Redis.from_url(redis_url)
        redis_conn.ping()
    except Exception:
        redis_mode = "redis-cli"
        try:
            redis_cli = _RedisCLI(redis_url)
            redis_cli.ping()
        except Exception as exc:
            print(
                "ERROR: Could not connect via redis-py or redis-cli. "
                "Install `redis` (pip) or ensure redis-cli is available.\n"
                f"Details: {exc}",
                file=sys.stderr,
            )
            return 2

    plan = _build_plan(args.queue, args.job_id)

    # Safety: staleness check based on meta.updated_at (if present).
    # This requires rq + redis-py. If we're running via redis-cli, we refuse unless forced.
    updated_at_raw: str | None = None
    updated_at_dt: datetime | None = None
    if redis_mode == "redis-py":
        updated_at_raw = _get_job_meta_updated_at(redis_conn, f"rq:job:{args.job_id}")
        updated_at_dt = _parse_iso8601(updated_at_raw) if updated_at_raw else None
        if args.min_stale_s > 0 and not args.force:
            if updated_at_dt is None:
                print(
                    "Refusing purge: could not read meta.updated_at for this job. "
                    "Re-run with --force if you are sure this is a zombie job.",
                    file=sys.stderr,
                )
                return 3
            age_s = int((_utcnow() - updated_at_dt).total_seconds())
            if age_s < args.min_stale_s:
                print(
                    f"Refusing purge: job meta.updated_at is only {age_s}s old (< {args.min_stale_s}s). "
                    "Re-run with --force if needed.",
                    file=sys.stderr,
                )
                return 3
    else:
        if args.min_stale_s > 0 and not args.force:
            print(
                "Refusing purge in redis-cli mode without --force. "
                "(Cannot read job meta to verify staleness.)",
                file=sys.stderr,
            )
            return 3

    dry_run = not args.execute

    print(f"Redis: {redis_url.split('@')[-1]}")
    print(f"Queue: {args.queue}")
    print(f"Job:   {args.job_id}")
    if updated_at_raw:
        print(f"meta.updated_at: {updated_at_raw}")
    print(f"Backend: {redis_mode}")
    print("Mode:  DRY-RUN" if dry_run else "Mode:  EXECUTE")

    # 1) Remove from queue list
    print(f"- LREM {plan.queue_key} {args.job_id}")
    if not dry_run:
        try:
            if redis_mode == "redis-py":
                redis_conn.lrem(plan.queue_key, 0, args.job_id)
            else:
                assert redis_cli is not None
                redis_cli.lrem(plan.queue_key, 0, args.job_id)
        except Exception as exc:
            print(f"  WARN: LREM failed: {exc}")

    # 2) Remove from registries
    for zkey in plan.registry_zset_keys:
        if redis_mode == "redis-py":
            members = _iter_zset_members(redis_conn, zkey)
        else:
            assert redis_cli is not None
            try:
                members = redis_cli.zrange(zkey, 0, -1)
            except Exception:
                members = []
        hits = _find_started_registry_members(members, args.job_id)
        if not hits:
            continue
        for member in hits:
            print(f"- ZREM {zkey} {member}")
            if not dry_run:
                try:
                    if redis_mode == "redis-py":
                        redis_conn.zrem(zkey, member)
                    else:
                        assert redis_cli is not None
                        redis_cli.zrem(zkey, member)
                except Exception as exc:
                    print(f"  WARN: ZREM failed for {zkey}: {exc}")

    # 3) Delete job keys
    for key in plan.job_keys:
        print(f"- DEL {key}")
    if not dry_run:
        try:
            if redis_mode == "redis-py":
                redis_conn.delete(*plan.job_keys)
            else:
                assert redis_cli is not None
                redis_cli.delete(*plan.job_keys)
        except Exception as exc:
            print(f"  WARN: DEL job keys failed: {exc}")

    # 4) Optionally remove worker record(s)
    worker_names: list[str] = []
    if args.worker:
        worker_names = [args.worker]
    elif args.clean_workers:
        if redis_mode == "redis-py":
            worker_names = _maybe_find_workers_for_job(redis_conn, plan, args.job_id)
        else:
            assert redis_cli is not None
            matches: list[str] = []
            try:
                worker_names_all = redis_cli.smembers(plan.worker_set_key)
            except Exception:
                worker_names_all = []
            for name in worker_names_all:
                if not name:
                    continue
                try:
                    h = redis_cli.hgetall(plan.worker_key_prefix + name)
                except Exception:
                    continue
                for candidate_field in ("current_job_id", "job_id", "current_job"):
                    if h.get(candidate_field) == args.job_id:
                        matches.append(name)
                        break
            worker_names = matches

    for w in worker_names:
        worker_key = plan.worker_key_prefix + w
        print(f"- SREM {plan.worker_set_key} {w}")
        print(f"- DEL {worker_key}")
        if not dry_run:
            try:
                if redis_mode == "redis-py":
                    redis_conn.srem(plan.worker_set_key, w)
                else:
                    assert redis_cli is not None
                    redis_cli.srem(plan.worker_set_key, w)
            except Exception as exc:
                print(f"  WARN: SREM worker failed: {exc}")
            try:
                if redis_mode == "redis-py":
                    redis_conn.delete(worker_key)
                else:
                    assert redis_cli is not None
                    redis_cli.delete(worker_key)
            except Exception as exc:
                print(f"  WARN: DEL worker failed: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
