"""Shared position sizing utility.

`compute_units` は元々 ``bb_mean_reversion.py`` のモジュールレベル関数として
存在していた. StrategyBase 抽象基底の導入に伴い, 戦略間で共通利用するため
本モジュールに切り出した. 仕様 (引数・戻り値・SKIP 条件) は同関数と完全同一.
"""
from __future__ import annotations

from math import floor


def compute_units(
    *,
    equity: float,
    atr: float,
    price: float,
    risk_pct: float,
    sl_atr_mult: float,
    margin: float,
    spread: float = 0.0,
) -> int:
    """Compute order units from risk, margin, ATR, and spread constraints.

    SKIP 条件を満たす場合は 0 を返す.

    ``units_margin_cap`` は backtesting.py の Broker と同様に spread 調整後価格
    ``price * (1 + |spread|)`` を基準とする (`backtesting.py:842/999` 参照).
    これを怠ると Strategy 側「発注可」の判断に対し Broker 側で無警告キャンセル
    (line 999-1002) が起き得る. ``spread=0.0`` は spread 未考慮の旧挙動.

    Returns:
        採用 units (正の整数). ``0`` は SKIP (機会見送り) を表す.

    SKIP 条件:
        - ``atr`` / ``price`` / ``margin`` が非正
        - ``units_risk > units_margin_cap`` (リスク量が証拠金制約を超過)
        - ``units_risk <= 0``
    """
    if atr <= 0 or price <= 0 or margin <= 0:
        return 0
    stop_distance = sl_atr_mult * atr
    if stop_distance <= 0:
        return 0
    units_risk = int(floor(equity * risk_pct / stop_distance))
    adjusted_price = price * (1 + abs(spread))
    units_margin_cap = int(floor(equity / margin / adjusted_price))
    if units_risk > units_margin_cap:
        return 0
    if units_risk <= 0:
        return 0
    return units_risk
