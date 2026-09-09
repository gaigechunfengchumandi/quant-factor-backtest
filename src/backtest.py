# -*- coding: utf-8 -*-
"""策略回测与因子检验。

实验一：黄金ETF 双均线趋势择时（MA20/MA120），对比买入持有基准。
        思路：趋势跟踪策略在趋势性资产上更有效（参数扫描结论见 README）。
实验二：ETF 篮子截面反转因子（120 日收益取负）IC/ICIR 检验 + Top-2 轮动回测。
        思路：A 股中周期截面呈反转效应（参数扫描结论见 README）。

防前视：T 日收盘出信号，T+1 日生效。
样本外：所有指标同时输出全样本与 2024-01-01 之后样本外两段。
指标口径：日频收益，年化 252 个交易日，无风险利率取 0，夏普 = 年化收益 / 年化波动。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "close_panel.csv")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

TRADING_DAYS = 252
OOS_START = "2024-01-01"  # 样本外起点


# ---------- 指标 ----------
def perf_metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    nav = (1 + ret).cumprod()
    ann_ret = nav.iloc[-1] ** (TRADING_DAYS / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    return {
        "annual_return": round(float(ann_ret), 4),
        "annual_vol": round(float(ann_vol), 4),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(dd), 4),
    }


def full_oos(ret: pd.Series) -> dict:
    return {
        "full_sample": perf_metrics(ret),
        "out_of_sample_2024+": perf_metrics(ret[ret.index >= OOS_START]),
    }


# ---------- 实验一：双均线趋势择时 ----------
def run_experiment_1(panel: pd.DataFrame, fast: int = 20, slow: int = 120) -> dict:
    close = panel["sh518880"].dropna()  # 黄金ETF
    df = pd.DataFrame({"close": close})
    df["ma_fast"] = df["close"].rolling(fast).mean()
    df["ma_slow"] = df["close"].rolling(slow).mean()
    df["signal"] = (df["ma_fast"] > df["ma_slow"]).astype(int)
    df["ret"] = df["close"].pct_change()
    df["strat_ret"] = df["signal"].shift(1) * df["ret"]  # T+1 生效，防前视
    df = df.dropna()

    nav = pd.DataFrame({
        f"双均线择时(MA{fast}/{slow})": (1 + df["strat_ret"]).cumprod(),
        "买入持有(黄金ETF)": (1 + df["ret"]).cumprod(),
    })
    ax = nav.plot(figsize=(9, 4.5), grid=True, alpha=0.9)
    ax.set_title("实验一：黄金ETF 双均线择时 vs 买入持有（净值）")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "exp1_equity_curve.png"), dpi=130)
    plt.close()

    return {
        "params": {"fast": fast, "slow": slow},
        "period": f"{df.index[0].date()} ~ {df.index[-1].date()}",
        "strategy": full_oos(df["strat_ret"]),
        "buy_hold": full_oos(df["ret"]),
    }


# ---------- 实验二：截面反转因子 ----------
def run_experiment_2(panel: pd.DataFrame, lookback: int = 120, horizon: int = 20, top_n: int = 2) -> dict:
    p = panel.drop(columns=["sh511880"]).dropna()  # 剔除货币ETF
    factor = -p.pct_change(lookback)               # 反转因子：跌幅大者得分高
    fwd = p.shift(-horizon) / p - 1

    ics = []
    for dt in factor.index[::horizon]:
        f, r = factor.loc[dt].dropna(), fwd.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) >= 4:
            ic, _ = stats.spearmanr(f[common], r[common])
            if np.isfinite(ic):
                ics.append((dt, ic))
    ic_s = pd.Series(dict(ics))
    ic_oos = ic_s[ic_s.index >= OOS_START]

    def ic_stats(s: pd.Series) -> dict:
        return {
            "ic_mean": round(float(s.mean()), 4),
            "icir": round(float(s.mean() / s.std()), 2) if s.std() > 0 else None,
            "ic_positive_ratio": round(float((s > 0).mean()), 4),
            "periods": int(len(s)),
        }

    daily_ret = p.pct_change()
    rebs = list(p.index[lookback::horizon])
    segs = []
    for i, dt in enumerate(rebs[:-1]):
        picks = factor.loc[dt].nlargest(top_n).index
        segs.append(daily_ret.loc[dt:rebs[i + 1], picks].mean(axis=1).iloc[1:])
    strat_ret = pd.concat(segs).sort_index()
    eq_ret = daily_ret.mean(axis=1).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(range(len(ic_s)), ic_s.values,
                color=["#16a34a" if v > 0 else "#dc2626" for v in ic_s], width=0.8)
    axes[0].set_title(f"反转因子 Rank IC 序列（均值 {ic_s.mean():.3f}）")
    axes[0].set_xticks([])
    nav = pd.DataFrame({
        f"反转Top{top_n}轮动": (1 + strat_ret).cumprod(),
        "篮子等权基准": (1 + eq_ret).cumprod(),
    })
    nav.plot(ax=axes[1], grid=True)
    axes[1].set_title("反转轮动 vs 等权基准（净值）")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "exp2_factor_ic.png"), dpi=130)
    plt.close()

    return {
        "params": {"lookback_days": lookback, "horizon_days": horizon, "top_n": top_n},
        "ic_full_sample": ic_stats(ic_s),
        "ic_out_of_sample_2024+": ic_stats(ic_oos),
        "rotation": full_oos(strat_ret),
        "equal_weight": full_oos(eq_ret),
    }


if __name__ == "__main__":
    panel = pd.read_csv(DATA, parse_dates=["date"], index_col="date")
    result = {
        "experiment_1_dual_ma_timing_gold": run_experiment_1(panel),
        "experiment_2_reversal_factor": run_experiment_2(panel),
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
