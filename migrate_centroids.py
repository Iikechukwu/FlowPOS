"""
SMART INVENTORY SYSTEM
One-time migration: backfill centroid_json for customers enrolled
before centroid support existed.

Safe to run multiple times — customers that already have a
centroid_json are skipped.

Usage:
    python migrate_centroids.py
"""

import json
import sqlite3
import sys

from face_db import DB_PATH, init_database, compute_centroid
from embedding_loader import load_embeddings


def migrate():
    print("=" * 50)
    print("   CENTROID MIGRATION")
    print("=" * 50)

    init_database()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT staff_id, name, embeddings_json, embedding_json, centroid_json
        FROM customers
    ''')
    rows = cursor.fetchall()

    if not rows:
        print("No customers in database — nothing to migrate.")
        conn.close()
        return

    updated = 0
    skipped_has_centroid = 0
    skipped_no_embeddings = 0

    for staff_id, name, embeddings_json, embedding_json, centroid_json in rows:

        if centroid_json:
            skipped_has_centroid += 1
            continue

        embeddings = []
        if embeddings_json:
            try:
                embeddings = [
                    __import__("numpy").array(e)
                    for e in json.loads(embeddings_json)
                ]
            except Exception:
                embeddings = []

        if not embeddings and embedding_json:
            try:
                embeddings = [
                    __import__("numpy").array(json.loads(embedding_json))
                ]
            except Exception:
                embeddings = []

        if not embeddings:
            print(f"  Skipping {name} ({staff_id}) — no embeddings stored.")
            skipped_no_embeddings += 1
            continue

        centroid = compute_centroid(embeddings)
        new_centroid_json = json.dumps(centroid.tolist())

        cursor.execute(
            "UPDATE customers SET centroid_json = ? WHERE staff_id = ?",
            (new_centroid_json, staff_id)
        )
        print(f"  Backfilled centroid for {name} ({staff_id}) "
              f"from {len(embeddings)} embedding(s).")
        updated += 1

    conn.commit()
    conn.close()

    print("\n" + "=" * 50)
    print(f"  Updated:                 {updated}")
    print(f"  Already had centroid:    {skipped_has_centroid}")
    print(f"  No embeddings (skipped): {skipped_no_embeddings}")
    print("=" * 50)

    if updated > 0:
        print("\nReloading embeddings into memory...")
        load_embeddings()
        print("Done. Restart the recognition module to pick up changes.")


if __name__ == "__main__":
    migrate()