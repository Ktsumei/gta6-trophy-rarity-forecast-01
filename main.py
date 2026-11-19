"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles. The score is the mean
pinball loss on the logit scale, so the program writes the quantiles it really
believes: a median that follows the trophy's own trajectory, and a spread that
widens with the horizon and with how far the median has to travel.

The model, per trophy, in logit space (logit(p) = ln(p / (100 - p))):

  1. Trajectory. Each read gives the trophy's rate and the players tracked.
     A rate is E/N: the players who earned it (E) over the players tracked
     (N). Both grow, and they pull in opposite directions: new players
     joining dilutes every rate, while players progressing through the game
     raises the ones they reach. The program fits a recency-weighted line to
     the logit's history against ln(1 + hours since the game released), so
     the recent direction of the rate -- after the last refresh, not the flat
     launch night -- is what it extrapolates, and it damps the slope so a
     short-run burst does not run away over a week.
  2. Dilution prior. When the history is flat (no movement to fit) and the
     horizon is short, a refresh is the likeliest next event: Exophase
     refreshes its counts and the tracked base jumps, which drops every
     rate but the one already at 100%. The prior drops the median for a
     short horizon and fades to nothing over a long one, where the fitted
     trajectory takes over.
  3. Spread. Wider than a plain random walk, because a refresh can move a
     rate several-fold and a week of play can move it more: a floor for
     model uncertainty, a term that grows with the horizon, and a term that
     grows with how far the median has travelled. A trophy already at 100%
     is pinned there with a narrow spread.

Standard library only, no network, deterministic, fast. Reads every *.in
file in the input directory and writes one *.out file per case.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL_Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# --- model constants (fixed in advance, never tuned on the practice) ---
# A rate is clamped to this band before its logit, matching the scorer.
RATE_FLOOR = 0.01
RATE_CEIL = 99.99
# The trend fit weights a read by how recent it is; a read a half-life old
# counts half. The launch night is flat and quantized, so the recent reads
# -- after the last refresh -- carry the direction.
TREND_HALF_LIFE_HOURS = 12.0
# The fitted slope is damped: a rate cannot keep its recent pace for a week.
DRIFT_CAP = 2.0
# The dilution prior: how far a rate falls when the tracked base jumps on a
# refresh, in logit units, and how fast that prior fades with the horizon.
DILUTION_LOGITS = 1.6
DILUTION_TAU_DAYS = 0.25
# The spread: a floor, a random-walk term per square-root day, and a term
# for how far the median travelled (a big move is an uncertain move).
SPREAD_FLOOR = 0.30
SPREAD_PER_SQRT_DAY = 0.45
SPREAD_PER_DRIFT = 0.35
# A trophy with no rate at all (never listed, or nobody has it yet) is
# anchored low, with a wide spread: a few players may earn it by the target.
UNLISTED_ANCHOR = 0.05
UNLISTED_SPREAD = 1.2
# A trophy at (essentially) 100% stays there; its spread is narrow.
PINNED_RATE = 99.5
PINNED_SPREAD = 0.15


def _clamp_rate(pct):
    return min(RATE_CEIL, max(RATE_FLOOR, pct))


def _logit(pct):
    p = _clamp_rate(pct)
    return math.log(p / (100.0 - p))


def _inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def _round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def _parse_time(text):
    """An ISO 8601 timestamp to an aware datetime; None when unparseable."""
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def _weighted_line(xs, ys, ws):
    """Weighted least squares for y = a + b*x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def _forecast_one(case, trophy_index):
    """Five quantiles for one trophy, in percent, non-decreasing."""
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    release = _parse_time(case.get("game", {}).get("release_date"))
    if release is None:
        release = _parse_time(history[0]["at"]) if history else None
    target = _parse_time(case["target"])

    # The anchor is the most recent read that actually lists the trophy;
    # a trophy nobody has yet, or one no read lists, is anchored low.
    anchor_rate = None
    for read in reversed(history):
        rate = read["rates"][trophy_index]
        if rate is not None and rate > 0:
            anchor_rate = rate
            break
    last_rate = anchor_rate if anchor_rate is not None else 0.0
    unlisted = anchor_rate is None

    # A trophy essentially everyone holds stays there.
    if last_rate >= PINNED_RATE:
        median = _logit(PINNED_RATE)
        sigma = PINNED_SPREAD
        return [_round4(_clamp_rate(_inv_logit(median + z * sigma)))
                for z in NORMAL_Z]

    # Gather the trophy's own series: (hours since release, logit(rate)).
    points = []
    for read in history:
        rate = read["rates"][trophy_index]
        if rate is None:
            continue
        moment = _parse_time(read["at"])
        if moment is None or release is None:
            continue
        hours = (moment - release).total_seconds() / 3600.0
        if hours < 0:
            hours = 0.0
        points.append((hours, _logit(rate)))

    # Fit the recent direction: a recency-weighted line of the logit
    # against ln(1 + hours). The launch night is flat, so it contributes
    # almost nothing once the base has refreshed.
    drift_trend = 0.0
    if len(points) >= 3 and target is not None and release is not None:
        last_hours = points[-1][0]
        xs = [math.log1p(h) for h, _ in points]
        ys = [y for _, y in points]
        ws = [0.5 ** ((last_hours - h) / TREND_HALF_LIFE_HOURS)
              for h, _ in points]
        fit = _weighted_line(xs, ys, ws)
        if fit is not None:
            _, slope = fit
            target_hours = (target - release).total_seconds() / 3600.0
            x_last = math.log1p(max(last_hours, 0.0))
            x_target = math.log1p(max(target_hours, 0.0))
            drift_trend = slope * (x_target - x_last)
            drift_trend = max(-DRIFT_CAP, min(DRIFT_CAP, drift_trend))

    # Dilution prior: with a flat history the likeliest next event is a
    # refresh, which jumps the tracked base and drops the rate. It only
    # applies when the fitted trend says almost nothing, and it fades over
    # a long horizon, where the trajectory takes over.
    drift_prior = 0.0
    if abs(drift_trend) < 0.3:
        drift_prior = -DILUTION_LOGITS * math.exp(
            -horizon_days / DILUTION_TAU_DAYS)

    drift = drift_trend + drift_prior
    if unlisted:
        # No rate to dilute or trend from: anchor low, spread wide.
        median = _logit(UNLISTED_ANCHOR)
        sigma = UNLISTED_SPREAD
    else:
        median = _logit(last_rate) + drift
        median = max(_logit(RATE_FLOOR), min(_logit(RATE_CEIL), median))
        sigma = math.sqrt(
            SPREAD_FLOOR ** 2
            + (SPREAD_PER_SQRT_DAY ** 2) * horizon_days
            + (SPREAD_PER_DRIFT * abs(drift)) ** 2)

    quantiles = [_round4(_clamp_rate(_inv_logit(median + z * sigma)))
                 for z in NORMAL_Z]
    # Rounding can only tie neighbours, never reverse them; enforce order.
    for j in range(1, len(quantiles)):
        if quantiles[j] < quantiles[j - 1]:
            quantiles[j] = quantiles[j - 1]
    return quantiles


def forecast_case(case, name):
    if list(case["levels"]) != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, "
                         f"not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rows.append({"id": trophy["id"],
                     "q": _forecast_one(case, i)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
