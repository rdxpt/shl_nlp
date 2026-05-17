from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class PrimaryMissingSlot(str, Enum):
    target_role = "target_role"
    seniority_level = "seniority_level"
    none = "none"


class FeatureTracker(BaseModel):
    is_jailbreak: bool
    out_of_scope: bool
    wants_comparison: bool
    comparison_targets: List[str]
    target_role: Optional[str] = None
    seniority_level: Optional[str] = None
    primary_missing_slot: PrimaryMissingSlot
    context_complete: bool
