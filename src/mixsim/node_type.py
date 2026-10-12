from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    ORIGIN = "origin"
    DESTINATION = "destination"
    SIGNALIZED = "signalized"
    UNSIGNALIZED = "unsignalized"


class TurnType(str, Enum):
    LEFT = "left"
    STRAIGHT = "straight"
    RIGHT = "right"
