# events.py


class Events:
    NO_FACE             = "NO_FACE"             # Nobody at shelf
    CUSTOMER_IDENTIFIED = "CUSTOMER_IDENTIFIED" # Known employee OR newly-created
                                                 # temp UNK- customer recognized —
                                                 # M4 doesn't need to tell them apart
    MULTIPLE_FACES      = "MULTIPLE_FACES"       # More than one face detected
    CUSTOMER_LEFT       = "CUSTOMER_LEFT"        # Face gone for 3+ seconds OR a different person detected mid-session


def build_event(event_type, customer_id=None,
                name=None, confidence=None,
                num_faces=0):
    """
    Standard payload that travels with every event.
    M4 receives this dictionary and acts on it.
    """
    return {
        "event":       event_type,
        "customer_id": customer_id,
        "name":        name,
        "confidence":  confidence,
        "num_faces":   num_faces
    }