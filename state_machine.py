"""Auditable lifecycle rules for CommunityConnect requests and connections."""

REQUEST_TRANSITIONS = {
    "open": {"matched", "cancelled", "expired"},
    "matched": {"closed", "cancelled", "open", "expired"},
    "expired": {"open"},
    "closed": set(),
    "cancelled": set(),
}

CONNECTION_TRANSITIONS = {
    "pending": {"accepted", "rejected", "withdrawn", "expired"},
    "accepted": {"withdrawn", "closed"},
    "rejected": set(),
    "withdrawn": set(),
    "expired": set(),
    "closed": set(),
}

TRANSITION_ACTORS = {
    ("request", "open", "matched"): {"system"},
    ("request", "open", "cancelled"): {"seeker", "admin"},
    ("request", "open", "expired"): {"system", "admin"},
    ("request", "matched", "closed"): {"seeker", "helper", "admin", "system"},
    ("request", "matched", "cancelled"): {"seeker", "admin"},
    ("request", "matched", "open"): {"helper", "system", "admin"},
    ("request", "matched", "expired"): {"system", "admin"},
    ("request", "expired", "open"): {"admin"},
    ("connection", "pending", "accepted"): {"helper", "admin"},
    ("connection", "pending", "rejected"): {"helper", "admin", "system"},
    ("connection", "pending", "withdrawn"): {"seeker", "admin"},
    ("connection", "pending", "expired"): {"system", "admin"},
    ("connection", "accepted", "withdrawn"): {"helper", "seeker", "admin"},
    ("connection", "accepted", "closed"): {"seeker", "helper", "admin", "system"},
}

STATUS_LABELS = {
    "request": {
        "open": "Looking for a helper",
        "matched": "Matched, waiting for helper to respond",
        "closed": "Completed",
        "cancelled": "Cancelled by you",
        "expired": "No response, request expired",
    },
    "connection": {
        "pending": "Waiting for your response",
        "accepted": "Confirmed",
        "rejected": "Not accepted",
        "withdrawn": "Helper withdrew, request reopened",
        "expired": "No response, offer expired",
        "closed": "Completed",
    },
}


def validate_transition(entity, current, target, actor):
    """Return a user-facing error, or None when the transition is valid."""
    if target not in (REQUEST_TRANSITIONS if entity == "request" else CONNECTION_TRANSITIONS).get(current, set()):
        return f"This {entity} is already {STATUS_LABELS[entity].get(current, current).lower()} and cannot be changed to {target}."
    if actor not in TRANSITION_ACTORS.get((entity, current, target), set()):
        return "You are not authorized to perform that action."
    return None


def status_label(entity, status, viewer="seeker"):
    if entity == "connection" and status == "pending":
        return "Waiting for your response" if viewer == "helper" else "Sent, awaiting reply"
    return STATUS_LABELS[entity].get(status, status)
