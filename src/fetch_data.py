# -*- coding: utf-8 -*-
"""拉取 A 股 ETF 日线行情（腾讯财经前复权 K 线接口），缓存为 CSV。

接口：https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
无需第三方数据 SDK，仅依赖 requests。
"""
import os
import time

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 跨资产 ETF 篮子：宽基 + 风格 + 商品 + 货币，保证截面分散度
ETF_UNIVERSE = {
    "sh510300": "沪深300ETF",
    "sh510500": "中证500ETF",
    "sh512100": "中证1000ETF",
    "sz159915": "创业板ETF",
    "sh588000": "科创50ETF",
    "sh518880": "黄金ETF",
    "sh511880": "银华日利ETF(货币)",
}

START = "2018-01-01"
END = "2026-09-09"
COUNT = 640  # 接口单次上限 640 根，按年分段拉取再拼接


def _fetch_range(symbol: str, start: str, end: str) -> list:
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={symbol},day,{start},{end},{COUNT},qfq")
    payload = requests.get(url, timeout=20).json()["data"][symbol]
    return payload.get("qfqday") or payload.get("day") or []


def fetch_one(symbol: str, name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    rows = []
    for year in range(int(START[:4]), int(END[:4]) + 1):
        s = f"{year}-01-01"
        e = min(f"{year}-12-31", END)
        rows.extend(_fetch_range(symbol, s, e))
        time.sleep(0.2)
    seen, uniq = set(), []
    for r in rows:  # 分段交界去重
        if r[0] not in seen:
            seen.add(r[0])
            uniq.append(r)
    df = pd.DataFrame([r[:6] for r in uniq],
                      columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = df[c].astype(float)
    df["symbol"] = symbol
    df["name"] = name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    time.sleep(0.3)
    return df


def load_panel() -> pd.DataFrame:
    """返回 date × symbol 的收盘价透视表，并缓存 close_panel.csv。"""
    frames = []
    for symbol, name in ETF_UNIVERSE.items():
        df = fetch_one(symbol, name)
        frames.append(df)
        print(f"[ok] {symbol} {name} rows={len(df)} {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    close = pd.concat(frames).pivot_table(index="date", columns="symbol", values="close")
    close = close.sort_index().ffill().dropna()
    close.to_csv(os.path.join(DATA_DIR, "close_panel.csv"), encoding="utf-8-sig")
    print(f"panel: {close.shape}, {close.index[0].date()} ~ {close.index[-1].date()}")
    return close


if __name__ == "__main__":
    load_panel()
