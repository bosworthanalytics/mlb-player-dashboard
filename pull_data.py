"""
MLB Player Comparison Data Pipeline
Jung Hoo Lee (SF Giants) vs Ceddanne Rafaela (BOS Red Sox)
Pulls: standard batting, Statcast advanced, plate discipline, defense
Output: 4 CSV files in output/ for Power BI import
"""

import os, time, warnings
import pandas as pd
import numpy as np
import requests
from datetime import date

warnings.filterwarnings('ignore')

try:
    from pybaseball import statcast_batter, batting_stats, cache
    cache.enable()
    PYBASEBALL_OK = True
except Exception as e:
    print(f"pybaseball import warning: {e}")
    PYBASEBALL_OK = False

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLAYERS = {
    "Jung Hoo Lee":    {"mlbam_id": 808982, "debut_season": 2024},
    "Ceddanne Rafaela": {"mlbam_id": 678882, "debut_season": 2023},
}

TODAY = date.today().isoformat()

SEASON_DATES = {
    2023: {"start": "2023-03-30", "end": "2023-10-01"},
    2024: {"start": "2024-03-20", "end": "2024-09-29"},
    2025: {"start": "2025-03-27", "end": "2025-09-28"},
    2026: {"start": "2026-03-27", "end": TODAY},
}

MONTH_NUM = {
    "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10,
}

SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "hit_into_play",
    "foul_bunt", "missed_bunt",
}
CONTACT_DESCS = {"foul", "foul_tip", "hit_into_play", "foul_bunt"}
IN_ZONE     = set(range(1, 10))   # zones 1-9
OUT_ZONE    = {11, 12, 13, 14}

# ── Helpers ────────────────────────────────────────────────────────────────────
def seasons_for(player):
    start = PLAYERS[player]["debut_season"]
    return [s for s in SEASON_DATES if s >= start]


def mlb_api(player_name, stat_type, season, group="hitting"):
    pid = PLAYERS[player_name]["mlbam_id"]
    url = (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
           f"?stats={stat_type}&season={season}&group={group}&gameType=R&sportId=1")
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    MLB API error ({player_name} {season} {stat_type}): {e}")
        return {}


_statcast_cache = {}

def get_statcast(player_name, season):
    key = (player_name, season)
    if key in _statcast_cache:
        return _statcast_cache[key]
    if not PYBASEBALL_OK:
        return pd.DataFrame()
    mlbam_id = PLAYERS[player_name]["mlbam_id"]
    sd = SEASON_DATES[season]
    print(f"    Fetching Statcast: {player_name} {season}...")
    try:
        df = statcast_batter(sd["start"], sd["end"], player_id=mlbam_id)
        _statcast_cache[key] = df
        time.sleep(1)
        return df
    except Exception as e:
        print(f"    Statcast error ({player_name} {season}): {e}")
        _statcast_cache[key] = pd.DataFrame()
        return pd.DataFrame()


# ── 1. BATTING STANDARD ────────────────────────────────────────────────────────
def pull_batting_standard():
    print("\n[1/4] Batting Standard Stats (MLB API monthly splits)...")
    rows = []

    NUM_TO_MONTH = {v: k for k, v in MONTH_NUM.items()}

    def stat_int(s, key):
        try: return int(s.get(key, 0) or 0)
        except: return 0

    def build_row(player, season, month_name, month_num, period, s):
        g   = stat_int(s, "gamesPlayed")
        pa  = stat_int(s, "plateAppearances")
        ab  = stat_int(s, "atBats")
        h   = stat_int(s, "hits")
        d   = stat_int(s, "doubles")
        t   = stat_int(s, "triples")
        hr  = stat_int(s, "homeRuns")
        rbi = stat_int(s, "rbi")
        sb  = stat_int(s, "stolenBases")
        cs  = stat_int(s, "caughtStealing")
        bb  = stat_int(s, "baseOnBalls")
        hbp = stat_int(s, "hitByPitch")
        sf  = stat_int(s, "sacFlies")
        so  = stat_int(s, "strikeOuts")
        tb  = h - d - t - hr + 2*d + 3*t + 4*hr
        avg = round(h/ab, 3)         if ab else 0.0
        obp = round((h+bb+hbp)/(ab+bb+hbp+sf), 3) if (ab+bb+hbp+sf) else 0.0
        slg = round(tb/ab, 3)        if ab else 0.0
        ops = round(obp+slg, 3)
        return {"Player": player, "Season": season, "Month": month_name,
                "Month_Num": month_num, "Period": period,
                "G": g, "PA": pa, "AB": ab, "H": h,
                "AVG": avg, "OBP": obp, "SLG": slg, "OPS": ops,
                "HR": hr, "RBI": rbi, "SB": sb, "CS": cs, "BB": bb, "SO": so}

    for player in PLAYERS:
        for season in seasons_for(player):
            # Use gameLog and aggregate by month
            data = mlb_api(player, "gameLog", season)
            splits = []
            try:
                splits = data["stats"][0]["splits"]
            except (KeyError, IndexError):
                pass

            # Group game-level splits by month
            from collections import defaultdict
            monthly = defaultdict(lambda: defaultdict(int))
            for sp in splits:
                s = sp.get("stat", {})
                game_date = sp.get("date", "")
                try:
                    mn = int(game_date[5:7])
                except Exception:
                    continue
                for key in ["gamesPlayed","plateAppearances","atBats","hits",
                            "doubles","triples","homeRuns","rbi","stolenBases",
                            "caughtStealing","baseOnBalls","hitByPitch",
                            "sacFlies","strikeOuts"]:
                    try: monthly[mn][key] += int(s.get(key, 0) or 0)
                    except: pass

            season_totals = defaultdict(int)
            for mn, s in sorted(monthly.items()):
                mname = NUM_TO_MONTH.get(mn, f"Month {mn}")
                rows.append(build_row(player, season, mname, mn, "Monthly", s))
                for k, v in s.items():
                    season_totals[k] += v

            if season_totals.get("atBats", 0) > 0:
                rows.append(build_row(player, season, "Season Total", 99, "Season", season_totals))

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "batting_standard.csv")
    df.to_csv(path, index=False)
    print(f"  [OK] Saved: {path}  ({len(df)} rows)")
    return df


# ── 2. STATCAST ADVANCED ───────────────────────────────────────────────────────
def agg_statcast_advanced(sc, player, season):
    """Aggregate pitch-level Statcast df into monthly advanced rows."""
    if sc.empty:
        return []

    sc = sc.copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"], errors="coerce")
    sc["month_num"] = sc["game_date"].dt.month
    sc["month_name"] = sc["game_date"].dt.strftime("%B")

    # BIP only for EV / LA / xBA / barrel
    bip = sc[sc["type"] == "X"].copy()
    rows = []

    for (mn, mname), grp in sc.groupby(["month_num", "month_name"]):
        bip_grp = bip[bip["month_num"] == mn]
        n_bip = len(bip_grp)

        ev_avg    = round(bip_grp["launch_speed"].mean(), 1)       if n_bip else None
        ev_max    = round(bip_grp["launch_speed"].max(), 1)        if n_bip else None
        la_avg    = round(bip_grp["launch_angle"].mean(), 1)       if n_bip else None
        xba       = round(bip_grp["estimated_ba_using_speedangle"].mean(), 3) if n_bip else None
        xslg      = round(bip_grp["estimated_slg_using_speedangle"].mean(), 3) if n_bip else None
        xwoba     = round(bip_grp["estimated_woba_using_speedangle"].mean(), 3) if n_bip else None
        barrel_ct = bip_grp["barrel"].sum()                        if "barrel" in bip_grp.columns and n_bip else 0
        barrel_p  = round(barrel_ct / n_bip * 100, 1)             if n_bip else None
        hh_ct     = (bip_grp["launch_speed"] >= 95).sum()         if n_bip else 0
        hh_p      = round(hh_ct / n_bip * 100, 1)                 if n_bip else None

        rows.append({
            "Player": player, "Season": season,
            "Month": mname, "Month_Num": mn, "Period": "Monthly",
            "xBA": xba, "xSLG": xslg, "xwOBA": xwoba,
            "EV_avg": ev_avg, "EV_max": ev_max, "LA_avg": la_avg,
            "Barrel_pct": barrel_p, "HardHit_pct": hh_p,
            "BIP": n_bip,
        })

    # Season total
    n_bip_s = len(bip)
    if n_bip_s:
        barrel_ct = bip["barrel"].sum() if "barrel" in bip.columns else 0
        hh_ct     = (bip["launch_speed"] >= 95).sum()
        rows.append({
            "Player": player, "Season": season,
            "Month": "Season Total", "Month_Num": 99, "Period": "Season",
            "xBA":  round(bip["estimated_ba_using_speedangle"].mean(), 3),
            "xSLG": round(bip["estimated_slg_using_speedangle"].mean(), 3),
            "xwOBA":round(bip["estimated_woba_using_speedangle"].mean(), 3),
            "EV_avg": round(bip["launch_speed"].mean(), 1),
            "EV_max": round(bip["launch_speed"].max(), 1),
            "LA_avg": round(bip["launch_angle"].mean(), 1),
            "Barrel_pct": round(barrel_ct / n_bip_s * 100, 1),
            "HardHit_pct": round(hh_ct / n_bip_s * 100, 1),
            "BIP": n_bip_s,
        })
    return rows


def fetch_sprint_speed():
    """Pull sprint speed from Baseball Savant for relevant seasons."""
    speeds = {}  # (player_name, season) -> sprint_speed
    mlbam_ids = {info["mlbam_id"]: name for name, info in PLAYERS.items()}

    for season in [2023, 2024, 2025, 2026]:
        url = (f"https://baseballsavant.mlb.com/leaderboard/sprint_speed"
               f"?min_competitive_runs=0&position=&team=&csv=true&season={season}")
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                id_col = next((c for c in df.columns if "player_id" in c.lower() or c.lower() == "id"), None)
                spd_col = next((c for c in df.columns if "speed" in c.lower()), None)
                if id_col and spd_col:
                    for _, row in df.iterrows():
                        pid = int(row[id_col]) if pd.notna(row[id_col]) else -1
                        if pid in mlbam_ids:
                            speeds[(mlbam_ids[pid], season)] = round(float(row[spd_col]), 1)
        except Exception as e:
            print(f"    Sprint speed fetch error ({season}): {e}")
        time.sleep(0.5)
    return speeds


def pull_statcast_advanced():
    print("\n[2/4] Statcast Advanced Metrics...")
    sprint_speeds = fetch_sprint_speed()
    rows = []

    for player in PLAYERS:
        for season in seasons_for(player):
            sc = get_statcast(player, season)
            month_rows = agg_statcast_advanced(sc, player, season)
            # Attach sprint speed to Season Total row
            sp = sprint_speeds.get((player, season))
            for r in month_rows:
                r["Sprint_speed"] = sp if r["Period"] == "Season" else None
            rows.extend(month_rows)

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "statcast_advanced.csv")
    df.to_csv(path, index=False)
    print(f"  [OK] Saved: {path}  ({len(df)} rows)")
    return df


# ── 3. PLATE DISCIPLINE ────────────────────────────────────────────────────────
def agg_plate_discipline(sc, player, season):
    """Derive plate discipline metrics from pitch-level Statcast data."""
    if sc.empty:
        return []

    sc = sc.copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"], errors="coerce")
    sc["month_num"] = sc["game_date"].dt.month
    sc["month_name"] = sc["game_date"].dt.strftime("%B")
    sc["zone"] = pd.to_numeric(sc.get("zone", pd.Series(dtype=float)), errors="coerce")
    sc["is_swing"]    = sc["description"].isin(SWING_DESCS)
    sc["is_contact"]  = sc["description"].isin(CONTACT_DESCS)
    sc["is_swstr"]    = sc["description"].isin({"swinging_strike", "swinging_strike_blocked"})
    sc["in_zone"]     = sc["zone"].isin(IN_ZONE)
    sc["out_zone"]    = sc["zone"].isin(OUT_ZONE)

    # PA-level K/BB from MLB API (more accurate), so we store pitch-level proxies here
    rows = []
    all_groups = list(sc.groupby(["month_num", "month_name"]))

    def calc_disc(grp):
        n = len(grp)
        if n == 0:
            return {}
        n_swing      = grp["is_swing"].sum()
        n_contact    = grp["is_contact"].sum()
        n_swstr      = grp["is_swstr"].sum()
        n_inzone     = grp["in_zone"].sum()
        n_outzone    = grp["out_zone"].sum()
        swing_inzone  = (grp["is_swing"] & grp["in_zone"]).sum()
        swing_outzone = (grp["is_swing"] & grp["out_zone"]).sum()
        cont_inzone   = (grp["is_contact"] & grp["in_zone"]).sum()
        cont_outzone  = (grp["is_contact"] & grp["out_zone"]).sum()

        return {
            "SwStr_pct":   round(n_swstr / n * 100, 1)            if n else None,
            "Zone_pct":    round(n_inzone / n * 100, 1)           if n else None,
            "Chase_pct":   round(swing_outzone / n_outzone * 100, 1) if n_outzone else None,
            "ZSwing_pct":  round(swing_inzone / n_inzone * 100, 1)   if n_inzone else None,
            "ZContact_pct":round(cont_inzone / swing_inzone * 100, 1) if swing_inzone else None,
            "OContact_pct":round(cont_outzone / swing_outzone * 100, 1) if swing_outzone else None,
            "Contact_pct": round(n_contact / n_swing * 100, 1)    if n_swing else None,
            "Pitches": n,
        }

    for (mn, mname), grp in all_groups:
        d = calc_disc(grp)
        rows.append({"Player": player, "Season": season,
                     "Month": mname, "Month_Num": mn, "Period": "Monthly",
                     "K_pct": None, "BB_pct": None,   # filled from MLB API below
                     **d})

    # Season total
    d = calc_disc(sc)
    rows.append({"Player": player, "Season": season,
                 "Month": "Season Total", "Month_Num": 99, "Period": "Season",
                 "K_pct": None, "BB_pct": None, **d})
    return rows


def fill_k_bb(disc_rows, batting_rows):
    """Fill K% and BB% into plate discipline from batting standard data."""
    lookup = {}
    for r in batting_rows:
        if r["PA"] and r["PA"] > 0:
            lookup[(r["Player"], r["Season"], r["Month"])] = {
                "K_pct": round(r["SO"] / r["PA"] * 100, 1),
                "BB_pct": round(r["BB"] / r["PA"] * 100, 1),
            }
    for r in disc_rows:
        key = (r["Player"], r["Season"], r["Month"])
        if key in lookup:
            r["K_pct"] = lookup[key]["K_pct"]
            r["BB_pct"] = lookup[key]["BB_pct"]
    return disc_rows


def pull_plate_discipline(batting_rows):
    print("\n[3/4] Plate Discipline (Statcast pitch-level aggregation)...")
    rows = []

    for player in PLAYERS:
        for season in seasons_for(player):
            sc = get_statcast(player, season)
            rows.extend(agg_plate_discipline(sc, player, season))

    rows = fill_k_bb(rows, batting_rows)
    df = pd.DataFrame(rows)
    col_order = ["Player","Season","Month","Month_Num","Period",
                 "K_pct","BB_pct","SwStr_pct","Chase_pct","ZSwing_pct",
                 "ZContact_pct","OContact_pct","Contact_pct","Zone_pct","Pitches"]
    df = df[[c for c in col_order if c in df.columns]]
    path = os.path.join(OUTPUT_DIR, "plate_discipline.csv")
    df.to_csv(path, index=False)
    print(f"  [OK] Saved: {path}  ({len(df)} rows)")
    return df


# ── 4. DEFENSE ─────────────────────────────────────────────────────────────────
def fetch_fg_fielding_api(season):
    """Pull UZR/DRS from FanGraphs new API endpoint."""
    url = (f"https://www.fangraphs.com/api/leaders/major-league/data"
           f"?age=&pos=all&stats=fld&lg=all&qual=0"
           f"&season={season}&season1={season}&month=0&hand=&team=0"
           f"&pageitems=500&pagenum=1&ind=0&rost=0&players=&type=1"
           f"&postseason=&sortdir=default&sortstat=UZR")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.fangraphs.com/leaders/major-league",
        "Accept": "application/json",
    }
    try:
        r = requests.get(url, timeout=15, headers=headers)
        if r.status_code == 200:
            j = r.json()
            # Response has 'data' key with list of player dicts
            data = j.get("data", j) if isinstance(j, dict) else j
            if isinstance(data, list) and len(data) > 0:
                return pd.DataFrame(data)
    except Exception as e:
        print(f"    FanGraphs fielding API error ({season}): {e}")
    return pd.DataFrame()


def pull_defense():
    print("\n[4/4] Defensive Metrics (OAA via pybaseball + FanGraphs UZR/DRS)...")
    from pybaseball import statcast_outs_above_average

    name_map = {
        808982: "Jung Hoo Lee",
        678882: "Ceddanne Rafaela",
    }

    # Last-name first lookups for FanGraphs (format: "Last, First")
    fg_name_map = {
        "lee, jung hoo":    "Jung Hoo Lee",
        "rafaela, ceddanne":"Ceddanne Rafaela",
        "rafaela, cedanne": "Ceddanne Rafaela",
    }

    rows = []
    for season in [2023, 2024, 2025, 2026]:
        print(f"  Season {season}...")

        # OAA from pybaseball (Baseball Savant)
        oaa_lookup = {}
        try:
            oaa_df = statcast_outs_above_average(season, "OF", min_att=10)
            for _, r in oaa_df.iterrows():
                pid = int(r.get("player_id", -1))
                if pid in name_map:
                    oaa_lookup[name_map[pid]] = {
                        "OAA": r.get("outs_above_average"),
                        "Fielding_runs": r.get("fielding_runs_prevented"),
                        "OAA_infront":  r.get("outs_above_average_infront"),
                        "OAA_behind":   r.get("outs_above_average_behind"),
                    }
        except Exception as e:
            print(f"    OAA error ({season}): {e}")

        # UZR/DRS from FanGraphs API
        fg_lookup = {}
        fg_df = fetch_fg_fielding_api(season)
        if not fg_df.empty:
            # Try to find name column
            name_col = next((c for c in fg_df.columns
                             if c.lower() in ("playername","name","playerid","-1")), None)
            if name_col is None and len(fg_df.columns) > 1:
                name_col = fg_df.columns[1]   # often second column is name
            if name_col:
                for _, fr in fg_df.iterrows():
                    n = str(fr.get(name_col, "")).lower().strip()
                    matched = fg_name_map.get(n)
                    if not matched:
                        for key, val in fg_name_map.items():
                            if key.split(",")[0] in n:
                                matched = val
                                break
                    if matched:
                        fg_lookup[matched] = {}
                        for col_key, col_names in {
                            "UZR":      ["UZR", "uzr"],
                            "UZR_150":  ["UZR/150", "uzr/150"],
                            "DRS":      ["DRS", "drs"],
                            "Arm_runs": ["ARM", "Arm", "arm"],
                            "Range_runs":["RngR", "rngr"],
                            "Error_runs":["ErrR", "errr"],
                        }.items():
                            for cn in col_names:
                                if cn in fg_df.columns:
                                    try:
                                        fg_lookup[matched][col_key] = round(float(fr[cn]), 1)
                                    except Exception:
                                        pass
                                    break

        for player in PLAYERS:
            if season < PLAYERS[player]["debut_season"]:
                continue
            oaa_data = oaa_lookup.get(player, {})
            fg_data  = fg_lookup.get(player, {})
            rows.append({
                "Player":        player,
                "Season":        season,
                "OAA":           oaa_data.get("OAA"),
                "Fielding_Runs": oaa_data.get("Fielding_runs"),
                "OAA_Infront":   oaa_data.get("OAA_infront"),
                "OAA_Behind":    oaa_data.get("OAA_behind"),
                "UZR":           fg_data.get("UZR"),
                "UZR_150":       fg_data.get("UZR_150"),
                "DRS":           fg_data.get("DRS"),
                "Arm_runs":      fg_data.get("Arm_runs"),
                "Range_runs":    fg_data.get("Range_runs"),
                "Error_runs":    fg_data.get("Error_runs"),
            })
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "defense.csv")
    df.to_csv(path, index=False)
    print(f"  [OK] Saved: {path}  ({len(df)} rows)")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MLB Data Pipeline: Lee vs. Rafaela")
    print(f"Run date: {TODAY}")
    print("=" * 60)

    batting_df = pull_batting_standard()
    batting_rows = batting_df.to_dict("records")

    statcast_df  = pull_statcast_advanced()
    disc_df      = pull_plate_discipline(batting_rows)
    defense_df   = pull_defense()

    print("\n" + "=" * 60)
    print("All done. Files saved to output/:")
    for f in ["batting_standard.csv", "statcast_advanced.csv",
              "plate_discipline.csv", "defense.csv"]:
        path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(path):
            sz = os.path.getsize(path)
            print(f"  [OK] {f}  ({sz:,} bytes)")
        else:
            print(f"  [MISSING] {f}  MISSING")
    print("=" * 60)
