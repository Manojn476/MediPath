from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SharedState:
    user_id: str
    language: str
    input_type: str
    intent: str
    active_branches: List[str] = field(default_factory=list)
    branch_outputs: Dict[str, Any] = field(default_factory=dict)
    new_data_flags: Dict[str, bool] = field(default_factory=lambda: {
        "adherence": False,
        "activity": False,
        "journal": False,
    })
    raw_input: Optional[str] = None
