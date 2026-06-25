# price_catalog.py
# This file contains all snack prices in Naira.

import datetime

PRICES = {
    "gala":            200,
    "cream_crackers":  200,
    "cheese_balls":    150,
    "nutri_yo":        600,
    "viju_milk":       1300,
    "fab":             500,
    "cabin_biscuit":   1200,
    "bottled_water":   300,
}
# NOTE: "bottled_water" confirmed as a real trained class by the
# person who trained the model (M3) — it's genuinely detectable,
# despite not appearing in confidence_filter.py's VALID_LABELS list.
# That list is stale/incomplete and unused elsewhere anyway (nothing
# in the live pipeline calls is_valid_detection()), so it isn't a
# reliable source for which classes the model actually knows.

def get_price(item_label):
    """
    Give this function an item name and it returns the price.
    If the item is not in the list it returns 0, warns you,
    and logs the unknown item to a file.
    """
    price = PRICES.get(item_label, 0)
    if price == 0:
        print(f"WARNING: '{item_label}' not in price catalog. Price set to 0.")
        with open("unknown_items_log.txt", "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] Unknown item detected: '{item_label}'\n")
    return price