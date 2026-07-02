from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import time

@dataclass
class Event:
    topic: str
    ts: float
    payload: Dict[str, Any]

class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable[[Event], None]]] = {}

    def subscribe(self, topic: str, cb: Callable[[Event], None]) -> None:
        self._subs.setdefault(topic, []).append(cb)

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        ev = Event(topic=topic, ts=time.time(), payload=payload)
        for cb in self._subs.get(topic, []):
            cb(ev)
