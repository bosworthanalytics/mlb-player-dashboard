# Rest-of-Season Projection Model — Project Scope

**Author:** Bosworth Analytics
**Goal:** Build a predictive model that projects each MLB player's **rest-of-season (ROS)** performance from data available at a mid-season cutoff — the project that moves the portfolio from *analytics* to *data science*.

**Decisions locked in:**
- **Scope:** Both hitters and pitchers (two pipelines)
- **Approach:** Explainable baseline → ML, and prove ML beats the baseline
- **Deliverable:** New **"Projections"** tab in the existing Streamlit dashboard

---

## 1. The modeling problem

A supervised **regression** problem framed around a cutoff date.

| | Hitters | Pitchers |
|---|---|---|
| **Target (y)** | ROS **wOBA** (secondary: OPS) | ROS **FIP** (secondary: ERA) |
| **Unit** | one (player, season, cutoff) row | one (player, season, cutoff) row |
| **Min playing time** | ≥150 PA pre-cutoff & ≥100 PA post | ≥40 IP pre-cutoff & ≥30 IP post |

**Why wOBA / FIP as targets:** both are computed from countable components (so they split cleanly by date) and are more stable / less luck-driven than AVG or ERA — the right thing to actually predict.

---

## 2. Data plan (the hard part — flagged early)

The core engineering challenge is the **pre/post-cutoff split**. Strategy:

1. **Standard components → MLB Stats API game logs** (`stats=gameLog`). Aggregate game-by-game into pre-cutoff and post-cutoff windows, then compute wOBA / FIP from raw components (BB, HBP, 1B/2B/3B/HR, AB, SF for wOBA; HR, BB, HBP, K, IP for FIP). No reliance on pre-split season tables.
2. **Statcast quality metrics → Baseball Savant**, date-filtered (EV, barrel%, hard-hit%, xwOBA, xwOBACON, whiff%, chase%; for pitchers: velo, movement, xwOBA-against). Pull pre-cutoff windows only for features. Cache aggressively.
3. **Prior-season & career baselines → FanGraphs / season tables** already wired into the app.
4. **Bio → MLB API** (age as of season, position, handedness — already have `get_player_career`).

**Training span:** 2021–2025 (full Statcast era, skip shortened 2020).
**Sample expansion:** use multiple cutoffs per season (e.g., May 31 / Jun 30 / Jul 31). Increases rows ~3×; note the within-season correlation as a limitation.

---

## 3. Features (inputs available *before* the cutoff only)

**Hitters**
- Pre-cutoff: wOBA, xwOBA, ISO, K%, BB%, chase%, whiff%, EV, barrel%, hard-hit%, sprint speed, PA
- History: prior-season wOBA, 3-yr weighted wOBA, career wOBA
- Context: age, position

**Pitchers**
- Pre-cutoff: FIP, xFIP, K%, BB%, SwStr%, fastball velo, avg movement, xwOBA-against, IP, GS share
- History: prior-season FIP, 3-yr weighted FIP, career FIP
- Context: age, role (SP/RP)

**Leakage guard:** every feature uses only pre-cutoff data; the target uses only post-cutoff. No exceptions.

---

## 4. Models (the "compare" story)

1. **Naive** — "ROS = pre-cutoff stat." The do-nothing benchmark.
2. **Marcel-style baseline** — weighted prior seasons (5/4/3) + current sample, regressed to league mean by PA/IP, with an age adjustment. The respected, fully explainable benchmark.
3. **Ridge regression** — linear, regularized; interpretable coefficients.
4. **Gradient boosting** (LightGBM or XGBoost) — the ML model; tuned via time-aware CV.

**Sample weighting:** weight rows by ROS PA/IP so noisy small-sample targets count less.

---

## 5. Validation (done honestly)

- **Temporal holdout:** train on 2021–2024, test on **2025** (never random-split — that leaks the future).
- **Metrics:** RMSE, MAE, R² for each model, plus correlation of projected vs. actual.
- **The headline result:** a table showing `Gradient Boosting < Ridge < Marcel < Naive` on holdout error.
- **Explainability:** SHAP (or gain) feature importance — shows *which* inputs drive ROS performance (expected: regression-to-mean signals + quality-of-contact dominate early-season surface stats).
- **Calibration check:** are extreme early-season hot/cold starts appropriately regressed?

**Success criterion:** ML model beats the Marcel baseline on 2025 holdout RMSE for both hitters and pitchers.

---

## 6. Dashboard integration — "Projections" tab

Model is trained **offline**; the app only loads artifacts (fast, no training in Streamlit).

Tab contents:
- **ROS projection leaderboard** — sortable by projected wOBA / FIP, filter hitters vs pitchers
- **Player drill-down** — prior season · current pre-cutoff · Marcel proj · ML proj · (actual, when available)
- **Backtest panel** — holdout RMSE/MAE vs baseline, projected-vs-actual scatter for 2025
- **Feature importance** chart (top drivers)
- A short "How this works / limitations" methodology blurb

Artifacts saved to `output/` (or a new `models/`):
`ros_projections_2026.csv`, `model_metrics.json`, `feature_importance.csv`, plus `*.joblib` model files.

---

## 7. Build phases (proposed order)

| Phase | Output |
|---|---|
| 1. Data collection | `pull_projection_data.py` → training table (player-season-cutoff rows) |
| 2. Feature engineering | clean feature matrix + target, leakage-audited |
| 3. Baseline | Marcel + naive benchmarks with holdout scores |
| 4. ML models | ridge + gradient boosting, tuned |
| 5. Validation | metrics table, feature importance, projected-vs-actual charts |
| 6. Generate 2026 projections | `ros_projections_2026.csv` + artifacts |
| 7. Dashboard tab | "Projections" tab wired into `app.py` |
| 8. Methodology write-up | `ROS_Projection_Methodology.md` for the portfolio/LinkedIn |

---

## 8. Tech stack & new dependencies

`pandas`, `numpy`, `scikit-learn` (ridge, metrics, CV), `lightgbm` (or `xgboost`), `shap`, `joblib`, plus existing `pybaseball` / `requests`. Add to `requirements.txt`.

---

## 9. Risks & honest limitations

- **Statcast pull volume** is the main cost — many players × seasons × date windows. Mitigate with Savant leaderboard CSV exports (season + date-filtered) rather than pitch-by-pitch where possible, and cache to disk.
- **Survivorship bias** — players who get hurt / demoted have no ROS target; results describe "players who kept playing." State this.
- **Multiple-cutoff rows are correlated** — acknowledge in methodology; optionally cluster-aware CV by player.
- **2026 is in progress** — current-season projections are live estimates; backtest is what proves the method.

---

## 10. Sources (per project rules)

FanGraphs, MLB.com / MLB Stats API, Baseball Savant (Statcast). Modeling: scikit-learn, LightGBM/XGBoost docs.
