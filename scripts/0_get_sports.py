"""
Download Chicago sports home game schedules for 2024-2026.
Uses ESPN's public API (no key needed).
Output: data/external/chicago_sports_games.csv

Teams near NNS:
  - Bulls (NBA) & Blackhawks (NHL): United Center (Near West Side, adjacent to NNS)
  - Cubs (MLB): Wrigley Field (Lake View, north of NNS)
  - Bears (NFL): Soldier Field (Near South Side)
  - White Sox (MLB): Rate Field (far south, less relevant but included)

Usage: python scripts/0_get_sports.py
"""

import json
import csv
import urllib.request
import time
from datetime import date

BASE = "https://site.api.espn.com/apis/site/v2/sports"

# ESPN team configs: (sport, league, team_id, team_name, venue)
TEAMS = [
    ("basketball", "nba", "4", "Bulls", "United Center"),
    ("hockey", "nhl", "4", "Blackhawks", "United Center"),
    ("baseball", "mlb", "16", "Cubs", "Wrigley Field"),
    ("baseball", "mlb", "4", "White Sox", "Rate Field"),
    ("football", "nfl", "3", "Bears", "Soldier Field"),
]

# Seasons to fetch
# NBA/NHL: season=2024 means 2023-24 season, season=2025 means 2024-25
# MLB/NFL: season=2024 means 2024 season
SEASONS = {
    "nba": [2024, 2025, 2026],   # 2023-24, 2024-25, 2025-26
    "nhl": [2024, 2025, 2026],   # 2023-24, 2024-25, 2025-26
    "mlb": [2024, 2025, 2026],
    "nfl": [2024, 2025],
}


def fetch_schedule(sport, league, team_id, season):
    """Fetch team schedule from ESPN API."""
    url = f"{BASE}/{sport}/{league}/teams/{team_id}/schedule?season={season}&seasontype=2"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def parse_games(data, team_name, venue):
    """Extract home games from ESPN schedule response."""
    games = []
    if not data:
        return games

    events = data.get("events", [])
    for event in events:
        try:
            # Get date
            event_date = event.get("date", "")[:10]
            if not event_date:
                continue

            # Check if home game
            competitions = event.get("competitions", [{}])
            if not competitions:
                continue

            comp = competitions[0]
            competitors = comp.get("competitors", [])

            is_home = False
            opponent = ""
            for c in competitors:
                team = c.get("team", {})
                home_away = c.get("homeAway", "")
                display = team.get("displayName", "") or team.get("shortDisplayName", "")

                if team_name.lower() in display.lower():
                    is_home = home_away == "home"
                elif home_away != "home":
                    opponent = display

            if not is_home:
                # Try alternative: check venue
                venue_info = comp.get("venue", {})
                venue_name = venue_info.get("fullName", "")
                if venue.lower() in venue_name.lower():
                    is_home = True

            if is_home:
                d = date.fromisoformat(event_date)
                # Only include games in our date range
                if date(2024, 1, 1) <= d <= date(2026, 6, 30):
                    games.append({
                        "date": event_date,
                        "team": team_name,
                        "opponent": opponent,
                        "venue": venue,
                        "league": "",  # filled in by caller
                    })
        except (KeyError, IndexError, ValueError):
            continue

    return games


def main():
    print("Downloading Chicago sports schedules from ESPN...\n")

    all_games = []

    for sport, league, team_id, team_name, venue in TEAMS:
        seasons = SEASONS.get(league, [2024, 2025])
        for season in seasons:
            label = f"{team_name} {league.upper()} {season}"
            print(f"  {label}...")
            data = fetch_schedule(sport, league, team_id, season)
            if data:
                games = parse_games(data, team_name, venue)
                for g in games:
                    g["league"] = league.upper()
                all_games.extend(games)
                print(f"    {len(games)} home games found")
            else:
                print(f"    No data returned")
            time.sleep(1)

    # Deduplicate by (date, team)
    seen = set()
    unique_games = []
    for g in all_games:
        key = (g["date"], g["team"])
        if key not in seen:
            seen.add(key)
            unique_games.append(g)

    unique_games.sort(key=lambda x: (x["date"], x["team"]))

    # Write output
    outfile = "data/external/chicago_sports_games.csv"
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "team", "league", "opponent", "venue"
        ])
        writer.writeheader()
        writer.writerows(unique_games)

    print(f"\nWrote {len(unique_games)} home games to {outfile}")

    # Summary by team
    print("\nHome games by team:")
    for team_name in ["Bulls", "Blackhawks", "Cubs", "White Sox", "Bears"]:
        count = sum(1 for g in unique_games if g["team"] == team_name)
        print(f"  {team_name}: {count}")

    # Summary by month
    print("\nHome games by month (all teams):")
    by_month = {}
    for g in unique_games:
        m = g["date"][:7]
        by_month[m] = by_month.get(m, 0) + 1
    for m in sorted(by_month.keys()):
        print(f"  {m}: {by_month[m]}")


if __name__ == "__main__":
    main()
