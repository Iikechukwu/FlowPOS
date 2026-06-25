"""
SMART INVENTORY SYSTEM
Entry point — Main Menu
"""

import sys
import os
import threading

from services.enrollment_manager import startup, show_system_info
from camera_stream import get_stream_url
from admin_panel import run_admin_panel

from enrollment.enrollment import run_enrollment
from face_db import get_all_customers, delete_customer
from embedding_loader import remove_embedding_from_memory

from m2.access_controller import AccessController
from station_bridge import StationBridge


# ── VIEW CUSTOMERS ────────────────────────────────────────────

def view_customers():
    customers = get_all_customers()
    if not customers:
        print("\nNo customers enrolled yet!")
        return

    print("\n" + "=" * 50)
    print("   ENROLLED CUSTOMERS")
    print("=" * 50)
    for c in customers:
        status  = "✅" if c["embedding"] is not None else "❌"
        unknown = " [UNKNOWN]" if c["staff_id"].startswith("UNK-") else ""
        print(f"\n{status} {c['name']}{unknown}")
        print(f"   ID       : {c['staff_id']}")
        print(f"   Enrolled : {c['enrolled_at']}")
        print(f"   Purchases: {c['total_purchases']}")
        print(f"   Spent    : ₦{c['total_spent']:.2f}")
        print("-" * 35)


# ── DELETE CUSTOMER ───────────────────────────────────────────

def delete_customer_menu():
    view_customers()
    staff_id = input("\nStaff ID to delete: ").strip()
    from face_db import customer_exists
    if not customer_exists(staff_id):
        print("Not found!")
        return
    confirm = input(f"Delete {staff_id}? (y/n): ").strip().lower()
    if confirm == "y":
        if delete_customer(staff_id):
            remove_embedding_from_memory(staff_id)
            print("Deleted successfully!")
    else:
        print("Cancelled!")


# ── LIVE RECOGNITION + BILLING ────────────────────────────────

def run_recognition(stream_url):
    """
    Starts live face recognition using the current enrolled
    database, wired to M4's billing/item-detection via
    StationBridge. Runs until Q is pressed or Ctrl+C.

    Needs a SECOND camera, separate from the face camera —
    one pointed at the shelf for item detection.
    """
    customers = get_all_customers()
    if not customers:
        print("\n⚠️  No customers enrolled!")
        print("Enroll at least one person first (option 1).")
        return

    print("\n" + "=" * 50)
    print("   LIVE RECOGNITION + BILLING")
    print("=" * 50)
    print(f"Enrolled customers: {len(customers)}")
    print("\nThis needs a SECOND camera, pointed at the shelf,")
    print("separate from the face camera you already connected.")

    item_stream_url = get_stream_url()
    model_path       = os.path.join("m4", "best.pt")

    bridge = StationBridge(
        item_model_path=model_path,
        item_camera_stream_url=item_stream_url,
        tripwire_line_x=420   # adjust once you see the live shelf feed
    )

    if not bridge.item_detection_available:
        print("\n⚠️  Continuing WITHOUT item detection/billing — "
              "only face recognition will run this session.")
        input("Press Enter to continue anyway, or Ctrl+C to cancel... ")

    item_thread = threading.Thread(
        target=bridge.run_item_detection_loop,
        daemon=True
    )
    item_thread.start()

    print("Press Q in the camera window or Ctrl+C to stop.\n")

    controller = AccessController(
        on_event=bridge.handle_m2_event,
        stream_url=stream_url
    )

    try:
        controller.run()
    finally:
        bridge.stop()
        print("\nRecognition + billing stopped.")


# ── MAIN MENU ─────────────────────────────────────────────────

def main():
    print("\n" + "=" * 50)
    print("   SMART INVENTORY SYSTEM")
    print("=" * 50)

    startup()
    stream_url = get_stream_url()

    while True:
        print("\n" + "=" * 50)
        print("   MAIN MENU")
        print("=" * 50)
        print("ENROLLMENT:")
        print("  1. Enrollment          (guided video · 17-step pipeline)")
        print()
        print("RECOGNITION:")
        print("  6. Live Recognition + Billing  (needs 2 cameras)")
        print()
        print("MANAGEMENT:")
        print("  2. View All Customers")
        print("  3. Delete Customer")
        print("  4. System Info")
        print("  5. Admin Panel")
        print()
        print("  0. Exit")
        print("=" * 50)

        choice = input("Enter choice (0-6): ").strip()

        if   choice == "1":
            run_enrollment(stream_url)
        elif choice == "2":
            view_customers()
        elif choice == "3":
            delete_customer_menu()
        elif choice == "4":
            show_system_info()
        elif choice == "5":
            run_admin_panel()
        elif choice == "6":
            run_recognition(stream_url)
        elif choice == "0":
            print("\nGoodbye! 👋")
            break
        else:
            print("Invalid choice — enter 0 to 6.")


if __name__ == "__main__":
    main()