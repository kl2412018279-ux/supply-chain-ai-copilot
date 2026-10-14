"""Rate card loading. Every rate lives in one JSON file (``RATES_PATH``) so a planner
can update diesel prices or contract rates without touching code."""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import get_settings


class RateCard:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    @classmethod
    def load(cls, path: str | Path | None = None) -> RateCard:
        path = Path(path or get_settings().rates_path)
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def with_overrides(
        self, diesel_rm_per_litre: float | None = None, margin_pct: float | None = None
    ) -> RateCard:
        """A what-if copy of the rate card (the original is left untouched)."""
        data = copy.deepcopy(self.data)
        if diesel_rm_per_litre is not None:
            data["fuel"]["diesel_rm_per_litre_peninsular"]["value"] = float(diesel_rm_per_litre)
        if margin_pct is not None:
            data["margin_pct"] = float(margin_pct)
        return RateCard(data)

    def trip_hours(self, destination: str) -> float:
        """Round-trip hours for one truck job to ``destination`` (drive + port turnaround)."""
        dest = self.destinations[self.resolve_destination(destination)]
        drive = 2 * dest["km"] / self.value("truck", "avg_speed_kmh")
        return drive + self.value("truck", "port_turnaround_hours")

    def value(self, section: str, key: str) -> float:
        entry = self.data[section][key]
        return float(entry["value"] if isinstance(entry, dict) else entry)

    def _public(self, section: str) -> dict[str, dict[str, Any]]:
        return {k: v for k, v in self.data[section].items() if not k.startswith("_")}

    @property
    def destinations(self) -> dict[str, dict[str, Any]]:
        return self._public("destinations")

    @property
    def origins(self) -> dict[str, dict[str, Any]]:
        return self._public("origins")

    @property
    def cargo_types(self) -> dict[str, dict[str, Any]]:
        return self._public("cargo")

    def container(self, size: str) -> dict[str, Any]:
        return self.data["container"][size]

    def resolve_destination(self, name: str) -> str:
        """Match ``'Penang'``, ``'KL'``, ``'Pasir Gudang, Johor'`` etc. to a destination key."""
        text = " " + " ".join(re.findall(r"[a-z]+", name.lower())) + " "
        for key, dest in self.destinations.items():
            phrases = [key.replace("-", " "), *dest.get("aliases", [])]
            phrases.append(" ".join(re.findall(r"[a-z]+", dest["label"].lower().split("(")[0])))
            if any(f" {p} " in text for p in phrases if p):
                return key
        known = ", ".join(d["label"] for d in self.destinations.values())
        raise KeyError(f"Unknown destination '{name}'. Known destinations: {known}")

    def find_destination(self, text: str) -> str | None:
        """The destination mentioned anywhere in free text, or ``None``."""
        try:
            return self.resolve_destination(text)
        except KeyError:
            return None


@lru_cache
def get_rate_card() -> RateCard:
    return RateCard.load()
