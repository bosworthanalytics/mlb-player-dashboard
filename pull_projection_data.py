"""
pull_projection_data.py  —  Phase 1 of the Rest-of-Season (ROS) projection model.

Builds a leakage-free training table of (player, season, cutoff) rows where:
  - features come ONLY from the pre-cutoff window
  - the target (wOBA for hitters, FIP for pitchers) comes ONLY from the post-cutoff window

Data source: MLB Stats API `byDateRange` at the league level (one request returns
every qualified player for a date window), so the whole pull is ~70 calls total.

Usage:
    py pull_projection_data.py            # all seasons (2021-2025)
    py pull_projection_data.py 2024       # single season (quick smoke test)

Outputs (to output/):
    proj_train_hitters.csv
    proj_train_pitchers.csv
    proj_league_constants.csv   (per-season FIP constant + league wOBA, for reference)
"""

import sys, time, math
from pathlib import Path
import requests
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
HEADERS   = {"User-Agent": "Mozilla/5.0 (BosworthAnalytics ROS model)"}
SEASONS   = [2021, 2022, 2023, 2024, 2025]
CUTOFFS   = {"0531": "05-31", "0630": "06-30", "0731": "07-31"}
SEASON_LO = "{}-03-01"   # generous bounds; byDateRange clips to actual games
SEASON_HI = "{}-11-15"

# Minimum playing time (tunable). Pre = enough signal; post = reliable target.
MIN_PRE_PA, MIN_POST_PA   = 100, 80
MIN_PRE_IP, MIN_POST_IP   = 25.0, 20.0

# wOBA linear weights (single canonical set — relative ordering is what the model
# learns; exact yearly weights vary by ~1-2%). Source: FanGraphs wOBA constants.
WOBA_W = {"bb": 0.69, "hbp": 0.72, "1b": 0.88, "2b": 1.24, "3b": 1.56, "hr": 2.00}

OUT = Path(__file__).resolve().parent / "output"
SLEEP = 0.34  # polite pause between API calls


# ── API helpers ──────────────────────────────────────────────────────────────
def _get(url):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                print(f"    ! request failed: {e}")
                return {}
            time.sleep(1.0)
    return {}


def fetch_range(season, group, start, end):
    """Return {mlbam: stat_dict(+name)} for all players over [start, end]."""
    url = (f"https://statsapi.mlb.com/api/v1/stats?stats=byDateRange&group={group}"
           f"&startDate={start}&endDate={end}&sportId=1&gameType=R&season={season}"
           f"&limit=5000&playerPool=All")
    splits = _get(url).get("stats", [{}])[0].get("splits", [])
    out = {}
    for sp in splits:
        pid = (sp.get("player") or {}).get("id")
        if pid is None:
            continue
        st = dict(sp.get("stat", {}))
        st["_name"] = (sp.get("player") or {}).get("fullName", "")
        # a player can appear on multiple teams in a window — accumulate
        if pid in out:
            out[pid] = _merge_stat(out[pid], st)
        else:
            out[pid] = st
    time.sleep(SLEEP)
    return out


def _merge_stat(a, b):
    """Sum countable fields across stints; keep name."""
    keys = ["plateAppearances","atBats","hits","doubles","triples","homeRuns",
            "baseOnBalls","intentionalWalks","hitByPitch","sacFlies","strikeOuts",
            "battersFaced","earnedRuns"]
    out = {"_name": a.get("_name") or b.get("_name")}
    for k in keys:
        out[k] = _num(a.get(k)) + _num(b.get(k))
    out["inningsPitched"] = _ip(a.get("inningsPitched")) + _ip(b.get("inningsPitched"))
    out["_ip_is_decimal"] = True
    return out


# ── Numeric helpers ──────────────────────────────────────────────────────────
def _num(v):
    try:
        return float(v) if v not in (None, "", "-") else 0.0
    except Exception:
        return 0.0


def _ip(v):
    """MLB IP is 'whole.outs' (e.g. 5.2 = 5 and 2/3). Returns decimal innings."""
    if isinstance(v, dict):
        return 0.0
    s = str(v)
    if s in ("", "None", "-"):
        return 0.0
    if "." in s:
        whole, frac = s.split(".")
        outs = int(frac) if frac else 0
        return int(whole) + outs / 3.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _ip_of(stat):
    return stat["inningsPitched"] if stat.get("_ip_is_decimal") else _ip(stat.get("inningsPitched"))


# ── Metric computations ──────────────────────────────────────────────────────
def hitter_metrics(s):
    pa  = _num(s.get("plateAppearances")); ab = _num(s.get("atBats"))
    h   = _num(s.get("hits")); d2 = _num(s.get("doubles")); d3 = _num(s.get("triples"))
    hr  = _num(s.get("homeRuns")); bb = _num(s.get("baseOnBalls"))
    ibb = _num(s.get("intentionalWalks")); hbp = _num(s.get("hitByPitch"))
    sf  = _num(s.get("sacFlies")); so = _num(s.get("strikeOuts"))
    if pa <= 0:
        return None
    b1  = h - d2 - d3 - hr
    num = (WOBA_W["bb"]*(bb-ibb) + WOBA_W["hbp"]*hbp + WOBA_W["1b"]*b1
           + WOBA_W["2b"]*d2 + WOBA_W["3b"]*d3 + WOBA_W["hr"]*hr)
    den = ab + bb - ibb + sf + hbp
    woba = num/den if den > 0 else None
    avg  = h/ab if ab > 0 else None
    obp  = (h+bb+hbp)/(ab+bb+hbp+sf) if (ab+bb+hbp+sf) > 0 else None
    slg  = (b1 + 2*d2 + 3*d3 + 4*hr)/ab if ab > 0 else None
    return {
        "PA": pa, "wOBA": woba, "AVG": avg, "OBP": obp,
        "ISO": (slg-avg) if (slg is not None and avg is not None) else None,
        "K%": so/pa*100, "BB%": bb/pa*100, "HR_rate": hr/pa*100,
    }


def pitcher_metrics(s, fip_const):
    ip  = _ip_of(s); tbf = _num(s.get("battersFaced"))
    if ip <= 0:
        return None
    hr  = _num(s.get("homeRuns")); bb = _num(s.get("baseOnBalls"))
    hbp = _num(s.get("hitByPitch")); so = _num(s.get("strikeOuts"))
    er  = _num(s.get("earnedRuns"))
    fip = (13*hr + 3*(bb+hbp) - 2*so)/ip + fip_const
    return {
        "IP": round(ip, 1), "FIP": fip, "ERA": er/ip*9,
        "K%": (so/tbf*100) if tbf > 0 else None,
        "BB%": (bb/tbf*100) if tbf > 0 else None,
        "HR9": hr/ip*9,
    }


def league_fip_constant(season):
    """FIP constant scales FIP to league ERA: lgERA - (13HR+3(BB+HBP)-2K)/IP."""
    full = fetch_range(season, "pitching", SEASON_LO.format(season), SEASON_HI.format(season))
    tHR=tBB=tHBP=tK=tER=0.0; tIP=0.0
    for s in full.values():
        tIP += _ip_of(s); tHR += _num(s.get("homeRuns")); tBB += _num(s.get("baseOnBalls"))
        tHBP += _num(s.get("hitByPitch")); tK += _num(s.get("strikeOuts"))
        tER += _num(s.get("earnedRuns"))
    if tIP <= 0:
        return 3.10, None, full
    lg_era = tER/tIP*9
    const  = lg_era - (13*tHR + 3*(tBB+tHBP) - 2*tK)/tIP
    return round(const, 3), round(lg_era, 3), full


def league_woba(full_hit):
    """League wOBA for reference."""
    agg = {}
    for s in full_hit.values():
        for k in ["plateAppearances","atBats","hits","doubles","triples","homeRuns",
                  "baseOnBalls","intentionalWalks","hitByPitch","sacFlies"]:
            agg[k] = agg.get(k, 0) + _num(s.get(k))
    m = hitter_metrics(agg)
    return round(m["wOBA"], 3) if m and m["wOBA"] else None


# ── Build training rows ──────────────────────────────────────────────────────
def build(seasons):
    hit_rows, pit_rows, const_rows = [], [], []
    # cache full prior-season metrics so each season's baseline is one lookup
    prior_hit_cache, prior_pit_cache = {}, {}

    def prior_full(season):
        """Full season-(N-1) per-player wOBA/FIP for the baseline feature."""
        if season in prior_hit_cache:
            return prior_hit_cache[season], prior_pit_cache[season]
        py = season - 1
        fc, _, pit_full = league_fip_constant(py)
        hit_full = fetch_range(py, "hitting", SEASON_LO.format(py), SEASON_HI.format(py))
        hm = {pid: hitter_metrics(s) for pid, s in hit_full.items()}
        pm = {pid: pitcher_metrics(s, fc) for pid, s in pit_full.items()}
        prior_hit_cache[season] = hm; prior_pit_cache[season] = pm
        return hm, pm

    for season in seasons:
        print(f"\n=== Season {season} ===")
        fip_const, lg_era, pit_full = league_fip_constant(season)
        hit_full = fetch_range(season, "hitting", SEASON_LO.format(season), SEASON_HI.format(season))
        lg_woba  = league_woba(hit_full)
        const_rows.append({"season": season, "fip_const": fip_const,
                           "lg_ERA": lg_era, "lg_wOBA": lg_woba})
        print(f"  league: FIP const={fip_const}  ERA={lg_era}  wOBA={lg_woba}")
        prior_hm, prior_pm = prior_full(season)

        for clabel, cmd in CUTOFFS.items():
            cutoff = f"{season}-{cmd}"
            # next day for the post window start
            post_start = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            pre_lo, pre_hi = SEASON_LO.format(season), cutoff
            post_hi = SEASON_HI.format(season)

            pre_h  = fetch_range(season, "hitting",  pre_lo,    cutoff)
            post_h = fetch_range(season, "hitting",  post_start, post_hi)
            pre_p  = fetch_range(season, "pitching", pre_lo,    cutoff)
            post_p = fetch_range(season, "pitching", post_start, post_hi)

            nh = np_ = 0
            for pid, s in pre_h.items():
                if pid not in post_h:
                    continue
                pre = hitter_metrics(s); post = hitter_metrics(post_h[pid])
                if not pre or not post or pre["wOBA"] is None or post["wOBA"] is None:
                    continue
                if pre["PA"] < MIN_PRE_PA or post["PA"] < MIN_POST_PA:
                    continue
                pr = prior_hm.get(pid)
                hit_rows.append({
                    "mlbam": pid, "name": s.get("_name", ""), "season": season, "cutoff": clabel,
                    "pre_PA": pre["PA"], "pre_wOBA": round(pre["wOBA"], 4),
                    "pre_AVG": _r(pre["AVG"]), "pre_OBP": _r(pre["OBP"]), "pre_ISO": _r(pre["ISO"]),
                    "pre_K%": _r(pre["K%"], 2), "pre_BB%": _r(pre["BB%"], 2),
                    "pre_HR_rate": _r(pre["HR_rate"], 2),
                    "prior_wOBA": _r(pr["wOBA"], 4) if pr and pr["wOBA"] else None,
                    "prior_PA": pr["PA"] if pr else None,
                    "target_wOBA": round(post["wOBA"], 4), "post_PA": post["PA"],
                }); nh += 1

            for pid, s in pre_p.items():
                if pid not in post_p:
                    continue
                pre = pitcher_metrics(s, fip_const); post = pitcher_metrics(post_p[pid], fip_const)
                if not pre or not post:
                    continue
                if pre["IP"] < MIN_PRE_IP or post["IP"] < MIN_POST_IP:
                    continue
                pr = prior_pm.get(pid)
                pit_rows.append({
                    "mlbam": pid, "name": s.get("_name", ""), "season": season, "cutoff": clabel,
                    "pre_IP": pre["IP"], "pre_FIP": round(pre["FIP"], 3), "pre_ERA": _r(pre["ERA"], 2),
                    "pre_K%": _r(pre["K%"], 2), "pre_BB%": _r(pre["BB%"], 2), "pre_HR9": _r(pre["HR9"], 2),
                    "prior_FIP": _r(pr["FIP"], 3) if pr and pr["FIP"] else None,
                    "prior_IP": pr["IP"] if pr else None,
                    "target_FIP": round(post["FIP"], 3), "post_IP": post["IP"],
                }); np_ += 1

            print(f"  cutoff {clabel}: {nh:4d} hitter rows, {np_:4d} pitcher rows")

    return (pd.DataFrame(hit_rows), pd.DataFrame(pit_rows), pd.DataFrame(const_rows))


def _r(v, n=3):
    return round(v, n) if v is not None else None


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    seasons = SEASONS
    if len(sys.argv) > 1:
        seasons = [int(a) for a in sys.argv[1:]]
        print(f"(quick run: seasons {seasons})")

    OUT.mkdir(exist_ok=True)
    hit_df, pit_df, const_df = build(seasons)

    hit_path = OUT / "proj_train_hitters.csv"
    pit_path = OUT / "proj_train_pitchers.csv"
    const_path = OUT / "proj_league_constants.csv"
    hit_df.to_csv(hit_path, index=False)
    pit_df.to_csv(pit_path, index=False)
    const_df.to_csv(const_path, index=False)

    print("\n--- Summary ------------------------------------------")
    print(f"  hitters : {len(hit_df):5d} rows  ->  {hit_path.name}")
    print(f"  pitchers: {len(pit_df):5d} rows  ->  {pit_path.name}")
    if not hit_df.empty:
        c = hit_df[["pre_wOBA","target_wOBA"]].corr().iloc[0,1]
        print(f"  hitter  pre_wOBA vs target_wOBA corr: {c:.3f}")
    if not pit_df.empty:
        c = pit_df[["pre_FIP","target_FIP"]].corr().iloc[0,1]
        print(f"  pitcher pre_FIP  vs target_FIP  corr: {c:.3f}")
    print("  (these naive correlations are the bar the model must beat)")


if __name__ == "__main__":
    main()
