"""hitl package — Human-in-the-Loop escalation and logging."""
from .escalation import log_escalation, get_all_escalations, mark_resolved

__all__ = ["log_escalation", "get_all_escalations", "mark_resolved"]
