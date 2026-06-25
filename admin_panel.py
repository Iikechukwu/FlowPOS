import cv2
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_db import (get_all_customers, customer_exists,
                     delete_customer, rename_unknown_customer)
from embedding_loader import (load_embeddings,
                               remove_embedding_from_memory)
from services.enrollment_manager import generate_staff_id

# ============================================
# GET ALL UNKNOWN CUSTOMERS
# ============================================

def get_unknown_customers():
    """
    Get all customers saved as UNKNOWN
    Returns list of unknown customers
    """
    customers = get_all_customers()
    unknowns = [c for c in customers
                if c["staff_id"].startswith("UNK-")]
    return unknowns

def get_known_customers():
    """
    Get all properly enrolled customers
    Returns list of known customers
    """
    customers = get_all_customers()
    known = [c for c in customers
             if not c["staff_id"].startswith("UNK-")]
    return known

# ============================================
# DISPLAY UNKNOWN CUSTOMER PHOTO
# ============================================

def show_unknown_photo(customer):
    """
    Display photo of unknown customer
    for manager to identify
    """
    folder_path = customer.get("folder_path", "")

    # Try to find best frame photo
    photo_path = os.path.join(folder_path, "best_frame.jpg")

    if not os.path.exists(photo_path):
        # Try other photo names
        for fname in ["preview.jpg", "fast_photo.jpg"]:
            alt_path = os.path.join(folder_path, fname)
            if os.path.exists(alt_path):
                photo_path = alt_path
                break

        # Try any jpg in folder
        if not os.path.exists(photo_path) and os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    photo_path = os.path.join(folder_path, f)
                    break

    if not os.path.exists(photo_path):
        print(f"No photo found for {customer['staff_id']}")
        return False

    # Load and display photo
    img = cv2.imread(photo_path)
    if img is None:
        print("Cannot read photo!")
        return False

    # Resize for display
    h, w = img.shape[:2]
    max_size = 400
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Add info overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 50), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    cv2.putText(img,
               f"ID: {customer['staff_id']}",
               (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 255, 255), 2)

    cv2.putText(img,
               f"Enrolled: {customer['enrolled_at']}",
               (10, 42),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (200, 200, 200), 1)

    cv2.putText(img,
               "Press any key to continue",
               (10, img.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.45, (150, 150, 150), 1)

    cv2.imshow(f"Unknown Customer: {customer['staff_id']}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True

# ============================================
# ASSIGN NAME TO UNKNOWN
# ============================================

def assign_name(unknown_customer, new_name, new_staff_id=None):
    """
    Assign a real name to an unknown (UNK-) customer.

    Unknown customers never have an embedding — they were never
    matched to anyone, that's why they're unknown. So this just
    renames the existing record (name + staff_id) in place; it does
    NOT add face recognition for this person. They will still show
    up as unknown next time they're at the shelf, until someone
    runs them through proper enrollment separately.

    Renaming in place (rather than delete-and-recreate) also means
    any purchases already logged against their temp UNK- ID stay
    attached to the renamed record instead of being lost.

    Returns True if successful.
    """
    try:
        old_staff_id = unknown_customer["staff_id"]

        if not new_staff_id:
            new_staff_id = generate_staff_id()

        success = rename_unknown_customer(
            old_staff_id=old_staff_id,
            new_name=new_name,
            new_staff_id=new_staff_id
        )

        if not success:
            print("Failed to assign name!")
            return False

        print(f"\n⚠️  Note: {new_name} is renamed but not yet")
        print(f"   enrolled for face recognition. Run enrollment")
        print(f"   separately so they're recognized next visit.")

        return True

    except Exception as e:
        print(f"Assignment error: {e}")
        return False

# ============================================
# VIEW ALL UNKNOWNS WITH PHOTOS
# ============================================

def review_unknown_customers():
    """
    Manager reviews all unknown customers
    Views photo and assigns name to each
    """
    print("\n" + "=" * 50)
    print("   UNKNOWN CUSTOMER REVIEW")
    print("=" * 50)

    unknowns = get_unknown_customers()

    if not unknowns:
        print("\nNo unknown customers found! ✅")
        print("All customers are properly enrolled")
        return

    print(f"\nFound {len(unknowns)} unknown customer(s)")
    print("Review each photo and assign a name\n")

    assigned = 0
    skipped = 0

    for i, customer in enumerate(unknowns):
        print("\n" + "-" * 40)
        print(f"Unknown {i+1}/{len(unknowns)}")
        print(f"ID: {customer['staff_id']}")
        print(f"Captured: {customer['enrolled_at']}")
        print("-" * 40)

        # Show photo
        print("\nOpening photo...")
        photo_shown = show_unknown_photo(customer)

        if not photo_shown:
            print("Could not show photo")

        # Manager options
        print("\nOptions:")
        print("1. Assign name to this person")
        print("2. Skip for now")
        print("3. Delete this unknown")
        print("4. Stop reviewing")

        choice = input("Choice (1/2/3/4): ").strip()

        if choice == "1":
            # Assign name
            name = input("Full Name: ").strip()
            if not name:
                print("Name required!")
                continue

            use_auto_id = input("Auto-generate Staff ID? (y/n): ").strip().lower()
            if use_auto_id == 'y':
                new_id = generate_staff_id()
                print(f"Auto ID: {new_id}")
            else:
                new_id = input("Enter Staff ID: ").strip()
                if not new_id:
                    new_id = generate_staff_id()

            success = assign_name(customer, name, new_id)
            if success:
                assigned += 1
                print(f"✅ Successfully assigned: {name}")
            else:
                print("❌ Assignment failed!")

        elif choice == "2":
            print("Skipped!")
            skipped += 1

        elif choice == "3":
            confirm = input(f"Delete {customer['staff_id']}? (y/n): ").strip().lower()
            if confirm == 'y':
                delete_customer(customer['staff_id'])
                remove_embedding_from_memory(customer['staff_id'])
                print("Deleted!")

        elif choice == "4":
            print("Stopped reviewing!")
            break

    print("\n" + "=" * 50)
    print("REVIEW COMPLETE!")
    print(f"Assigned: {assigned}")
    print(f"Skipped: {skipped}")
    print(f"Remaining unknowns: {len(unknowns) - assigned}")
    print("=" * 50)

# ============================================
# VIEW ALL CUSTOMERS
# ============================================

def view_all_customers():
    """View all customers — known and unknown"""
    customers = get_all_customers()

    if not customers:
        print("\nNo customers enrolled yet!")
        return

    known = [c for c in customers
             if not c["staff_id"].startswith("UNK-")]
    unknowns = [c for c in customers
                if c["staff_id"].startswith("UNK-")]

    print("\n" + "=" * 50)
    print("   ALL CUSTOMERS")
    print("=" * 50)

    print(f"\n✅ ENROLLED CUSTOMERS ({len(known)}):")
    print("-" * 35)
    for c in known:
        emb = "✅" if c["embedding"] is not None else "❌"
        print(f"  {emb} {c['name']}")
        print(f"     ID: {c['staff_id']}")
        print(f"     Enrolled: {c['enrolled_at']}")
        print(f"     Purchases: {c['total_purchases']}")
        print(f"     Spent: ₦{c['total_spent']:.2f}")

    if unknowns:
        print(f"\n⚠️  UNKNOWN CUSTOMERS ({len(unknowns)}):")
        print("-" * 35)
        for c in unknowns:
            print(f"  ❓ {c['staff_id']}")
            print(f"     Captured: {c['enrolled_at']}")
            print("     Name not assigned yet")

    print("\n" + "=" * 50)
    print(f"Total enrolled: {len(known)}")
    print(f"Total unknown: {len(unknowns)}")
    print(f"Total customers: {len(customers)}")
    print("=" * 50)

# ============================================
# ADMIN PANEL MENU
# ============================================

def run_admin_panel():
    """Main admin panel menu"""
    print("\n" + "=" * 50)
    print("   SMART INVENTORY - ADMIN PANEL")
    print("=" * 50)

    # Load embeddings
    load_embeddings()

    while True:
        # Check unknowns
        unknowns = get_unknown_customers()
        unknown_count = len(unknowns)

        print("\n" + "=" * 50)
        print("   ADMIN MENU")
        if unknown_count > 0:
            print(f"   ⚠️  {unknown_count} unknown customer(s) need attention!")
        print("=" * 50)
        print("1. Review Unknown Customers")
        print("2. View All Customers")
        print("3. Exit Admin Panel")
        print("=" * 50)

        choice = input("Choice (1/2/3): ").strip()

        if choice == "1":
            review_unknown_customers()
        elif choice == "2":
            view_all_customers()
        elif choice == "3":
            print("Exiting admin panel!")
            break
        else:
            print("Invalid choice!")

# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    run_admin_panel()