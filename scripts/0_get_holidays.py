"""
Generate comprehensive holidays/events CSV for Chicago demand analysis.
Includes US federal, religious, cultural, and Chicago-specific events.
Output: data/external/holidays_chicago.csv

Usage: python scripts/0_get_holidays.py
"""

import csv
from datetime import date, timedelta


def easter_date(year):
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def nth_weekday(year, month, weekday, n):
    """Find the nth occurrence of a weekday in a month. weekday: 0=Mon, 6=Sun."""
    first = date(year, month, 1)
    first_occ = first + timedelta(days=(weekday - first.weekday()) % 7)
    return first_occ + timedelta(weeks=n - 1)


def last_weekday(year, month, weekday):
    """Find the last occurrence of a weekday in a month."""
    if month == 12:
        last_day = date(year, 12, 31)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def main():
    events = []

    for year in [2024, 2025, 2026]:

        # ============================================================
        # US FEDERAL HOLIDAYS
        # ============================================================
        events.append((date(year, 1, 1), "New Year's Day", "us_federal", "negative"))
        events.append((nth_weekday(year, 1, 0, 3), "MLK Day", "us_federal", "negative"))
        events.append((nth_weekday(year, 2, 0, 3), "Presidents Day", "us_federal", "negative"))
        events.append((last_weekday(year, 5, 0), "Memorial Day", "us_federal", "negative"))
        events.append((date(year, 6, 19), "Juneteenth", "us_federal", "mixed"))
        events.append((date(year, 7, 4), "July 4th", "us_federal", "negative"))
        events.append((nth_weekday(year, 9, 0, 1), "Labor Day", "us_federal", "negative"))
        events.append((nth_weekday(year, 10, 0, 2), "Columbus Day", "us_federal", "neutral"))
        events.append((date(year, 11, 11), "Veterans Day", "us_federal", "neutral"))
        # Thanksgiving: 4th Thursday of November
        tg = nth_weekday(year, 11, 3, 4)
        events.append((tg, "Thanksgiving", "us_federal", "negative"))
        events.append((tg + timedelta(days=1), "Black Friday", "us_commercial", "negative"))
        events.append((tg + timedelta(days=2), "Thanksgiving Weekend", "us_commercial", "negative"))
        events.append((date(year, 12, 25), "Christmas", "us_federal", "negative"))

        # Adjacent days (known demand impact)
        events.append((date(year, 12, 24), "Christmas Eve", "us_commercial", "negative"))
        events.append((date(year, 12, 26), "Day After Christmas", "us_commercial", "negative"))
        events.append((date(year, 12, 31), "New Year's Eve", "us_commercial", "mixed"))
        events.append((date(year, 1, 2), "Day After NY", "us_commercial", "negative"))

        # Pre-holiday suppression (identified in analysis)
        events.append((tg - timedelta(days=1), "Pre-Thanksgiving Wed", "demand_pattern", "negative"))
        events.append((tg - timedelta(days=2), "Pre-Thanksgiving Tue", "demand_pattern", "negative"))

        # ============================================================
        # CULTURAL / COMMERCIAL
        # ============================================================
        events.append((date(year, 2, 14), "Valentine's Day", "commercial", "positive"))
        events.append((date(year, 10, 31), "Halloween", "commercial", "mixed"))

        # Mother's Day: 2nd Sunday of May
        events.append((nth_weekday(year, 5, 6, 2), "Mother's Day", "commercial", "mixed"))
        # Father's Day: 3rd Sunday of June
        events.append((nth_weekday(year, 6, 6, 3), "Father's Day", "commercial", "neutral"))

        # ============================================================
        # CHRISTIAN HOLIDAYS
        # ============================================================
        e = easter_date(year)
        events.append((e - timedelta(days=2), "Good Friday", "christian", "negative"))
        events.append((e, "Easter Sunday", "christian", "negative"))

        # ============================================================
        # JEWISH HOLIDAYS (dates shift annually on Gregorian calendar)
        # ============================================================

    # Jewish holidays — manually specified (Hebrew calendar conversion)
    jewish = [
        # Rosh Hashanah (2 days)
        (date(2024, 10, 2), "Rosh Hashanah", "jewish", "negative"),
        (date(2024, 10, 3), "Rosh Hashanah Day 2", "jewish", "negative"),
        (date(2025, 9, 22), "Rosh Hashanah", "jewish", "negative"),
        (date(2025, 9, 23), "Rosh Hashanah Day 2", "jewish", "negative"),
        (date(2026, 9, 11), "Rosh Hashanah", "jewish", "negative"),
        (date(2026, 9, 12), "Rosh Hashanah Day 2", "jewish", "negative"),
        # Yom Kippur
        (date(2024, 10, 12), "Yom Kippur", "jewish", "negative"),
        (date(2025, 10, 1), "Yom Kippur", "jewish", "negative"),
        (date(2026, 9, 21), "Yom Kippur", "jewish", "negative"),
        # Passover (first 2 days)
        (date(2024, 4, 22), "Passover", "jewish", "negative"),
        (date(2024, 4, 23), "Passover Day 2", "jewish", "negative"),
        (date(2025, 4, 12), "Passover", "jewish", "negative"),
        (date(2025, 4, 13), "Passover Day 2", "jewish", "negative"),
        (date(2026, 4, 1), "Passover", "jewish", "negative"),
        (date(2026, 4, 2), "Passover Day 2", "jewish", "negative"),
        # Hanukkah (first day)
        (date(2024, 12, 25), "Hanukkah Start", "jewish", "neutral"),  # overlaps Christmas
        (date(2025, 12, 14), "Hanukkah Start", "jewish", "neutral"),
    ]
    events.extend(jewish)

    # ============================================================
    # CHINESE NEW YEAR
    # ============================================================
    events.append((date(2024, 2, 10), "Chinese New Year", "chinese", "mixed"))
    events.append((date(2025, 1, 29), "Chinese New Year", "chinese", "mixed"))
    events.append((date(2026, 2, 17), "Chinese New Year", "chinese", "mixed"))

    # ============================================================
    # CHICAGO-SPECIFIC EVENTS (confirmed recurring, >20% demand impact)
    # ============================================================

    # St Patrick's Day — Chicago's parade is Saturday before Mar 17
    for year in [2024, 2025, 2026]:
        mar17 = date(year, 3, 17)
        # Find Saturday on or before Mar 17
        days_to_sat = (mar17.weekday() - 5) % 7
        parade_sat = mar17 - timedelta(days=days_to_sat)
        events.append((parade_sat, "St Patrick's Parade Chicago", "chicago", "positive"))
        events.append((mar17, "St Patrick's Day", "irish", "positive"))
        # Day after parade also elevated
        events.append((parade_sat + timedelta(days=1), "St Patrick's Sunday", "chicago", "positive"))

    # Lollapalooza (Grant Park, usually late July / early Aug, 4 days Thu-Sun)
    events.append((date(2024, 8, 1), "Lollapalooza", "chicago", "positive"))
    events.append((date(2024, 8, 2), "Lollapalooza", "chicago", "positive"))
    events.append((date(2024, 8, 3), "Lollapalooza", "chicago", "positive"))
    events.append((date(2024, 8, 4), "Lollapalooza", "chicago", "positive"))
    events.append((date(2025, 7, 31), "Lollapalooza", "chicago", "positive"))
    events.append((date(2025, 8, 1), "Lollapalooza", "chicago", "positive"))
    events.append((date(2025, 8, 2), "Lollapalooza", "chicago", "positive"))
    events.append((date(2025, 8, 3), "Lollapalooza", "chicago", "positive"))

    # Chicago Marathon (October Sunday)
    events.append((date(2024, 10, 13), "Chicago Marathon", "chicago", "mixed"))
    events.append((date(2025, 10, 12), "Chicago Marathon", "chicago", "mixed"))

    # NASCAR Chicago Street Race (Grant Park)
    events.append((date(2024, 7, 6), "NASCAR Chicago", "chicago", "mixed"))
    events.append((date(2024, 7, 7), "NASCAR Chicago", "chicago", "mixed"))
    events.append((date(2025, 7, 5), "NASCAR Chicago", "chicago", "mixed"))
    events.append((date(2025, 7, 6), "NASCAR Chicago", "chicago", "mixed"))

    # DNC 2024 (one-time, Aug 19-22)
    events.append((date(2024, 8, 19), "DNC 2024", "chicago", "positive"))
    events.append((date(2024, 8, 20), "DNC 2024", "chicago", "positive"))
    events.append((date(2024, 8, 21), "DNC 2024", "chicago", "positive"))
    events.append((date(2024, 8, 22), "DNC 2024", "chicago", "positive"))

    # ============================================================
    # WRITE OUTPUT
    # ============================================================

    # Filter to our date range
    events = [(d, n, c, e) for d, n, c, e in events
              if date(2024, 1, 1) <= d <= date(2026, 6, 30)]
    events.sort(key=lambda x: x[0])

    outfile = "data/external/holidays_chicago.csv"
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "event_name", "category", "expected_impact"])
        for d, name, cat, impact in events:
            writer.writerow([d.isoformat(), name, cat, impact])

    print(f"Wrote {len(events)} events to {outfile}")

    # Summary by category
    cats = {}
    for _, _, c, _ in events:
        cats[c] = cats.get(c, 0) + 1
    print("\nEvents by category:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
