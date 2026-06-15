from __future__ import annotations


def activity_payload(activity: str) -> dict:
    return {"activity": activity, "backend": "manual"}

