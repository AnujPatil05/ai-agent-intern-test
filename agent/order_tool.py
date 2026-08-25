"""
agent/order_tool.py

Order lookup tool.  Raw orders.json NEVER enters the model context.
Only the sanitized result of lookup_order() is passed to the LLM.

Privacy rules (enforced here, not by the LLM):
  - customer.name / customer.email / customer.shipping_address  → always stripped
  - Everything inside 'internal' (risk_score, warehouse_note, support_tags) → always stripped
  - carrier / tracking_number / estimated_delivery suppressed when status is
    'cancelled' or 'returned' (those fields may be stale operational remnants)

Status precedence (enforced here, not by the LLM):
  - 'status' field is authoritative
  - estimated_delivery=None + status='shipped' → report shipped, say no estimate
  - status='exception' → sets requires_handoff=True in result
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ORDERS_PATH = Path(__file__).parent.parent / "data" / "orders.json"

# Whitelist: the only fields that may reach the LLM
_SAFE_ORDER_FIELDS = {
    "order_id", "membership_tier", "placed_at", "status",
    "status_updated_at", "shipped_at", "delivered_at",
    "carrier", "tracking_number", "estimated_delivery",
    "customer_safe_message",
}

# Fields that are stale and misleading when an order is cancelled or returned
_STALE_WHEN_INACTIVE = {"carrier", "tracking_number", "estimated_delivery"}

# Statuses where carrier/ETA data must be suppressed
_INACTIVE_STATUSES = {"cancelled", "returned"}

# Safe item sub-fields
_SAFE_ITEM_FIELDS = {"name", "quantity", "final_sale"}

# Compiled pattern for valid order IDs after normalization
_ORDER_ID_RE = re.compile(r"^ORD-\d+$")

# Module-level cache — loaded once per process
_cache: dict[str, dict] | None = None
_snapshot_at: str = ""


def _load() -> tuple[dict[str, dict], str]:
    global _cache, _snapshot_at
    if _cache is None:
        raw = json.loads(_ORDERS_PATH.read_text(encoding="utf-8"))
        _snapshot_at = raw.get("snapshot_at", "")
        _cache = {o["order_id"]: o for o in raw.get("orders", [])}
        logger.debug("orders loaded: %d records, snapshot_at=%s", len(_cache), _snapshot_at)
    return _cache, _snapshot_at


def normalize_order_id(raw: str) -> str | None:
    """
    Normalize a user-supplied order ID string.

    Accepts:  'ord-1007', ' ORD-1007 ', 'ORD1007' (with space instead of dash).
    Rejects:  completely different strings, empty input.
    Returns None when the value cannot be a valid order ID.
    """
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip().upper()
    # Allow internal whitespace in place of dash: "ORD 1007" → "ORD-1007"
    cleaned = re.sub(r"[\s]+", "-", cleaned)
    if _ORDER_ID_RE.match(cleaned):
        return cleaned
    return None


def _sanitize(raw: dict, suppress_stale: bool) -> dict:
    """Return only safe fields from a raw order record."""
    result: dict[str, Any] = {}
    for field in _SAFE_ORDER_FIELDS:
        if field not in raw:
            continue
        value = raw[field]
        if suppress_stale and field in _STALE_WHEN_INACTIVE:
            value = None  # suppress misleading carrier/ETA
        result[field] = value

    result["items"] = [
        {k: item[k] for k in _SAFE_ITEM_FIELDS if k in item}
        for item in raw.get("items", [])
    ]
    return result


def lookup_order(order_id: str) -> dict[str, Any]:
    """
    Look up an order and return a sanitized, privacy-safe result dict.

    The LLM must not call this function unless it has an order_id.
    Application code (agent.py) enforces that gate; this function
    trusts that the caller has already validated presence of an ID.

    Return shapes
    -------------
    Found:
        {
          "found": True,
          "order": {<safe fields>},
          "stale_fields_suppressed": bool,
          "requires_handoff": bool,
          "snapshot_at": str,
        }
    Not found:
        {"found": False, "reason": "not_found",  "order_id": str}
    Malformed:
        {"found": False, "reason": "malformed",  "raw_input": str}
    """
    logger.info("lookup_order called | raw_id=%r", order_id)

    normalized = normalize_order_id(order_id)
    if normalized is None:
        logger.warning("lookup_order: malformed id | raw=%r", order_id)
        return {"found": False, "reason": "malformed", "raw_input": str(order_id)}

    orders, snapshot_at = _load()
    raw = orders.get(normalized)

    if raw is None:
        logger.warning("lookup_order: not found | id=%s", normalized)
        return {"found": False, "reason": "not_found", "order_id": normalized}

    status = raw.get("status", "").lower()
    suppress_stale = status in _INACTIVE_STATUSES
    requires_handoff = status == "exception"

    sanitized = _sanitize(raw, suppress_stale)

    result = {
        "found": True,
        "order": sanitized,
        "stale_fields_suppressed": suppress_stale,
        "requires_handoff": requires_handoff,
        "snapshot_at": snapshot_at,
    }
    logger.info(
        "lookup_order result | id=%s status=%s suppress_stale=%s handoff=%s",
        normalized, status, suppress_stale, requires_handoff,
    )
    return result


def get_snapshot_at() -> str:
    """Return the dataset snapshot timestamp (used as 'now' for policy math)."""
    _, ts = _load()
    return ts
