# core/rules.py
from __future__ import annotations
import json
from typing import Dict, Tuple

DEFAULT_PRESETS = {
    "short": {
        "logic": "AND",
        "min_pass": 2,
        "scoring": {"mode": "weighted_sum", "pass_threshold": 2.5, "normalize": False},
        "rules": [
            {"id":"ADX", "op":">=", "value":20, "length":14, "weight":1.0, "required":True},
            {"id":"+DI>-DI", "op":"bull", "length":14, "weight":0.7},
            {"id":"SMA_X", "op":"sma_gt", "fast":50, "slow":200, "weight":0.8},
            {"id":"RSI", "op":"between", "low":45, "high":65, "length":14, "weight":0.5},
            {"id":"STOCH", "op":"crossesUp", "min_level":55, "k_len":14, "d_len":3, "weight":0.5},
            {"id":"NEAR_52W", "op":">=", "value":0.97, "weight":0.5},
            {"id":"ATR%", "op":"between", "low":0.01, "high":0.08, "length":14, "weight":0.3}
        ]
    },
    "mid": {
        "logic": "AND",
        "min_pass": 3,
        "scoring": {"mode": "weighted_sum", "pass_threshold": 3.0, "normalize": False},
        "rules": [
            {"id":"ADX", "op":">=", "value":25, "length":14, "weight":1.0, "required":True},
            {"id":"+DI>-DI", "op":"bull", "length":14, "weight":0.7},
            {"id":"SMA_X", "op":"sma_gt", "fast":50, "slow":200, "weight":0.8, "required":True},
            {"id":"RSI", "op":"between", "low":50, "high":70, "length":14, "weight":0.5},
            {"id":"MACD_HIST", "op":">", "value":0.0, "weight":0.5},
            {"id":"NEAR_52W", "op":">=", "value":0.98, "weight":0.5},
            {"id":"ATR%", "op":"between", "low":0.01, "high":0.06, "length":14, "weight":0.3}
        ]
    },
    "long": {
        "logic": "AND",
        "min_pass": 3,
        "scoring": {"mode": "weighted_sum", "pass_threshold": 3.0, "normalize": False},
        "rules": [
            {"id":"ADX", "op":">=", "value":25, "length":14, "weight":1.0, "required":True},
            {"id":"+DI>-DI", "op":"bull", "length":14, "weight":0.7},
            {"id":"SMA_X", "op":"sma_gt", "fast":50, "slow":200, "weight":0.8, "required":True},
            {"id":"RSI", "op":"between", "low":45, "high":65, "length":14, "weight":0.5},
            {"id":"NEAR_52W", "op":">=", "value":0.99, "weight":0.5},
            {"id":"RET_6M", "op":">", "value":0.0, "weight":0.4},
            {"id":"ATR%", "op":"between", "low":0.005, "high":0.05, "length":14, "weight":0.3}
        ]
    }
}

def preset_json(name: str) -> str:
    return json.dumps(DEFAULT_PRESETS.get(name, DEFAULT_PRESETS["mid"]), indent=2)

def _cmp(value, op, ref_low=None, ref_high=None):
    if value is None or (isinstance(value, float) and (np.isnan(value))):
        return False
    try:
        if op in (">", "gt"):   return value >  ref_low
        if op in ("<", "lt"):   return value <  ref_low
        if op in (">=",):       return value >= ref_low
        if op in ("<=",):       return value <= ref_low
        if op == "between":     return (value >= ref_low) and (value <= ref_high)
        if op == "==":          return value == ref_low
        return False
    except Exception:
        return False

import numpy as np
def eval_rule(rule: Dict, m: Dict) -> tuple[bool, float]:
    rid = rule.get("id"); op = rule.get("op"); w = float(rule.get("weight", 0.0))
    passed = False
    if rid == "ADX":
        passed = _cmp(m.get("adx"), op, rule.get("value"))
    elif rid == "+DI>-DI":
        passed = bool(m.get("di_bias_bull", False))
    elif rid == "SMA_X":
        passed = m.get("sma_fast", float("-inf")) > m.get("sma_slow", float("inf"))
    elif rid == "RSI":
        passed = _cmp(m.get("rsi"), op, rule.get("low"), rule.get("high")) if op=="between" else _cmp(m.get("rsi"), op, rule.get("value"))
    elif rid == "MACD_HIST":
        passed = _cmp(m.get("macd_hist"), op, rule.get("value"))
    elif rid == "STOCH":
        if op == "crossesUp":
            passed = bool(m.get("stoch_cross_up", False)) and (m.get("stoch_k", 0) >= rule.get("min_level", 50))
        elif op == "crossesDown":
            passed = bool(m.get("stoch_cross_down", False)) and (m.get("stoch_k", 100) <= rule.get("max_level", 50))
    elif rid == "MFI":
        passed = _cmp(m.get("mfi"), op, rule.get("low"), rule.get("high")) if op=="between" else _cmp(m.get("mfi"), op, rule.get("value"))
    elif rid == "NEAR_52W":
        passed = _cmp(m.get("near_52w_high"), op, rule.get("value"))
    elif rid == "ATR%":
        passed = _cmp(m.get("atr_pct"), op, rule.get("low"), rule.get("high")) if op=="between" else _cmp(m.get("atr_pct"), op, rule.get("value"))
    elif rid == "RET_3M":
        passed = _cmp(m.get("ret_3m"), op, rule.get("value"))
    elif rid == "RET_6M":
        passed = _cmp(m.get("ret_6m"), op, rule.get("value"))
    elif rid == "RET_12M":
        passed = _cmp(m.get("ret_12m"), op, rule.get("value"))
    return bool(passed), (w if passed else 0.0)

def eval_ruleset(ruleset: Dict, metrics: Dict) -> Dict:
    logic = ruleset.get("logic", "AND").upper()
    min_pass = int(ruleset.get("min_pass", 0))
    rules = ruleset.get("rules", [])
    required_ok = True; pass_count = 0; score = 0.0

    # tek geçiş
    results = []
    for r in rules:
        p, s = eval_rule(r, metrics)
        results.append((p, s, r.get("required", False)))
        if r.get("required", False) and not p:
            required_ok = False
        pass_count += int(p); score += s

    if logic == "AND":
        all_pass = all(p for (p, _, req) in results if not req) and required_ok
    else:
        all_pass = any(p for (p, _, req) in results if not req) and required_ok

    if min_pass > 0:
        all_pass = (pass_count >= min_pass) and required_ok

    pass_threshold = float(ruleset.get("scoring", {}).get("pass_threshold", 0))
    score_pass = (score >= pass_threshold)
    final_pass = all_pass and score_pass
    return {"pass": final_pass, "score": score, "pass_count": pass_count, "required_ok": required_ok}

