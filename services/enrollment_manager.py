import os
import json
import time
import cv2
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_db import (init_database, save_customer,
                     get_all_customers, customer_exists,
                     get_database_info)
from embedding_loader import (load_embeddings,
                               add_embedding_to_memory,
                               get_embedding_count)

# CONFIGURATION

CUSTOMERS_DIR = "customers"
TEMP_DIR = "temp_enrollment"


# CUSTOMER FOLDER MANAGEMENT


def create_customer_folder(name):
    """Create folder for customer photos"""
    folder_name = name.replace(" ", "_").lower()
    folder_path = os.path.join(CUSTOMERS_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path, folder_name


# SAVE ENROLLMENT PROGRESS


def save_progress(staff_id, name, folder_path,
                  photos_captured, photo_paths):
    """Save enrollment progress for resume"""
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        progress = {
            "staff_id": staff_id,
            "name": name,
            "folder_path": folder_path,
            "photos_captured": photos_captured,
            "photo_paths": photo_paths,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        temp_path = os.path.join(TEMP_DIR, f"{staff_id}_progress.json")
        with open(temp_path, 'w') as f:
            json.dump(progress, f, indent=4)
        return temp_path
    except Exception as e:
        print(f"Progress save error: {e}")
        return None

def load_progress(staff_id):
    """Load saved enrollment progress"""
    try:
        temp_path = os.path.join(TEMP_DIR, f"{staff_id}_progress.json")
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Progress load error: {e}")
        return None

def clear_progress(staff_id):
    """Clear progress after successful enrollment"""
    try:
        temp_path = os.path.join(TEMP_DIR, f"{staff_id}_progress.json")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Progress clear error: {e}")

def check_pending_enrollments():
    """Check for interrupted enrollments"""
    try:
        if not os.path.exists(TEMP_DIR):
            return []
        return [f for f in os.listdir(TEMP_DIR)
                if f.endswith('_progress.json')]
    except:
        return []

def generate_staff_id():
    """
    Auto generate unique staff ID
    Format: SI-001, SI-002, SI-003...
    """
    try:
        customers = get_all_customers()
        
        if not customers:
            return "SI-001"
        
        # Get all existing IDs
        existing_ids = [c["staff_id"] for c in customers]
        
        # Find highest number
        max_num = 0
        for id in existing_ids:
            try:
                # Extract number from SI-XXX format
                if id.startswith("SI-"):
                    num = int(id.split("-")[1])
                    if num > max_num:
                        max_num = num
            except:
                continue
        
        # Generate next ID
        next_num = max_num + 1
        new_id = f"SI-{next_num:03d}"
        
        print(f"Generated Staff ID: {new_id}")
        return new_id
        
    except Exception as e:
        print(f"ID generation error: {e}")
        # Fallback to timestamp
        return f"SI-{int(time.time())}"
    


# ============================================
# SYSTEM STARTUP
# ============================================

def startup():
    """Initialize system at startup"""
    print("=" * 50)
    print("   SMART INVENTORY - STARTUP")
    print("=" * 50)

    # Initialize database
    init_database()

    # Load embeddings into memory
    count = load_embeddings()
    print(f"Embeddings loaded: {count}")

    # Check pending enrollments
    pending = check_pending_enrollments()
    if pending:
        print(f"\n⚠️  {len(pending)} interrupted enrollment(s) found!")
        print("Select enrollment option to resume")

    print("System ready!\n")
    return count

# ============================================
# SYSTEM INFO
# ============================================

def show_system_info():
    """Show system status"""
    get_database_info()
    print(f"Embeddings in memory: {get_embedding_count()}")
    pending = check_pending_enrollments()
    print(f"Pending enrollments: {len(pending)}")

# ============================================
# TEST ENROLLMENT MANAGER
# ============================================

if __name__ == "__main__":
    startup()
    show_system_info()