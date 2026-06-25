# ============================================
# SMART INVENTORY SYSTEM
# Integration Test — P1 + P2
# Tests face recognition pipeline
# ============================================

import sys
import os
from camera_stream import get_stream_url
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding_loader import load_embeddings
from m2.events import Events
from m2.access_controller import AccessController

# ============================================
# EVENT HANDLER — receives events from P2
# ============================================

def on_event(event):
    """
    Handles events fired by P2's AccessController
    This is where P4 billing will connect later
    """
    event_type = event["event"]
    name = event.get("name", "Unknown")
    confidence = event.get("confidence", 0)
    customer_id = event.get("customer_id", None)

    print("\n" + "=" * 50)

    if event_type == Events.CUSTOMER_IDENTIFIED:
        if customer_id and str(customer_id).startswith("UNK-"):
            print(f"❓ UNKNOWN PERSON — TEMP SESSION OPENED")
            print(f"   Temp ID: {customer_id}")
            print(f"   → Photo saved for manager review")
            print(f"   → Session started under temp ID!")
            print(f"   → P4 billing activated!")
        else:
            print(f"✅ CUSTOMER IDENTIFIED!")
            print(f"   Name: {name}")
            print(f"   ID: {customer_id}")
            print(f"   Confidence: {confidence}")
            print("   → Session started!")
            print("   → P4 billing activated!")

    elif event_type == Events.CUSTOMER_LEFT:
        print(f"👋 CUSTOMER LEFT!")
        print(f"   Name: {name}")
        print(f"   → Session ended!")
        print(f"   → P4 calculates bill!")

    elif event_type == Events.MULTIPLE_FACES:
        num = event.get("num_faces", 0)
        print(f"⚠️  MULTIPLE FACES: {num} people")
        print(f"   → One person at a time please")

    elif event_type == Events.NO_FACE:
        pass  # Don't print — happens every frame

    print("=" * 50)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("   P1 + P2 INTEGRATION TEST")
    print("=" * 50)

    # Load all enrolled embeddings
    print("\nLoading face database...")
    count = load_embeddings()
    print(f"Loaded {count} enrolled customers")

    if count == 0:
        print("\n⚠️  No customers enrolled!")
        print("Please enroll customers first using main.py")
        sys.exit(1)

    print(f"\n✅ Database ready with {count} customers")
    print("Starting face recognition...\n")

    # Start P2's access controller
    # Get DroidCam stream URL
    stream_url = get_stream_url()

    # Start P2's access controller with stream URL
    controller = AccessController(on_event=on_event, stream_url=stream_url)
    controller.run()