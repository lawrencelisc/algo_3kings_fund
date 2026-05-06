"""
HYPE Funding Rate vs Log Return — Correlation Analysis
=======================================================
功能：
  1. 從 Hyperliquid API 下載 HYPE 過去半年的
     - Hourly funding rate history
     - Hourly OHLC K 線
  2. 對齊數據，計算 log return
  3. Winsorize 去除極端值（1–99%）
  4. 計算三種 correlation：
     - 同期：funding[t] vs return[t]
     - 預測：funding[t] vs return[t+1]   ← 最重要，無 look-ahead bias
     - 累積：cumulative funding vs cumulative return
  5. 輸出：
     - hype_aligned.csv        原始對齊數據
     - hype_correlation.csv    三種 correlation 結果
     - hype_charts.png         三張圖

使用方法：
  pip install requests pandas matplotlib scipy
  python hype_funding_analysis.py
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import time
from datetime import datetime, timezone, timedelta

# ── 設定 ─────────────────────────────────────────────────────────────
COIN         = "HYPE"
MONTHS_BACK  = 6
WINSORIZE_LO = 0.01    # 1%
WINSORIZE_HI = 0.99    # 99%
API_URL      = "https://api.hyperliquid.xyz/info"
# ─────────────────────────────────────────────────────────────────────


def hl_post(payload: dict, retries: int = 3) -> list | dict:
    """POST to Hyperliquid info endpoint with retry."""
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt+1}/{retries}: {e}")
            time.sleep(2)


def fetch_funding_history(coin: str, start_ms: int) -> pd.DataFrame:
    """Download all hourly funding records from start_ms to now."""
    print(f"下載 {coin} funding history...")
    all_records = []
    cursor = start_ms

    while True:
        data = hl_post({"type": "fundingHistory", "coin": coin, "startTime": cursor})
        if not data:
            break
        all_records.extend(data)
        last_t = data[-1]["time"]
        if last_t <= cursor or len(data) < 2:
            break
        cursor = last_t + 1
        if last_t > int(datetime.now(timezone.utc).timestamp() * 1000):
            break
        time.sleep(0.2)

    df = pd.DataFrame(all_records)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["premium"]     = df["premium"].astype(float)
    df = df.sort_values("time").reset_index(drop=True)
    print(f"  → {len(df)} 條記錄")
    return df


def fetch_candles(coin: str, start_ms: int) -> pd.DataFrame:
    """Download hourly OHLC candles."""
    print(f"下載 {coin} 小時 K 線...")
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    data = hl_post({
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": "1h", "startTime": start_ms, "endTime": end_ms}
    })
    df = pd.DataFrame(data)
    df.columns = ["time_open", "open", "high", "low", "close", "volume",
                  "time_close", "n_trades"] if len(df.columns) == 8 else df.columns

    # Hyperliquid candle field names: t, o, h, l, c, v, T, n
    rename = {"t": "time_open", "o": "open", "h": "high",
              "l": "low",  "c": "close", "v": "volume"}
    df = df.rename(columns=rename)
    df["time_open"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("time_open").reset_index(drop=True)
    print(f"  → {len(df)} 條 K 線")
    return df


def winsorize(series: pd.Series, lo: float, hi: float) -> pd.Series:
    lo_val = series.quantile(lo)
    hi_val = series.quantile(hi)
    return series.clip(lo_val, hi_val)


def align_data(funding_df: pd.DataFrame, candle_df: pd.DataFrame) -> pd.DataFrame:
    """Align funding rate with candle close price by hour."""
    # Round both to hour
    funding_df = funding_df.copy()
    candle_df  = candle_df.copy()
    funding_df["hour"] = funding_df["time"].dt.floor("h")
    candle_df["hour"]  = candle_df["time_open"].dt.floor("h")

    # Keep one funding record per hour (last one if duplicates)
    funding_hr = funding_df.groupby("hour").last().reset_index()[["hour", "fundingRate", "premium"]]

    # Compute log return from candles
    candle_df["log_return"] = np.log(candle_df["close"] / candle_df["close"].shift(1))
    candle_hr = candle_df[["hour", "close", "log_return"]].copy()

    # Merge
    df = pd.merge(candle_hr, funding_hr, on="hour", how="inner")
    df = df.dropna(subset=["log_return", "fundingRate"]).reset_index(drop=True)

    # Convert to % for readability
    df["funding_rate_pct"] = df["fundingRate"] * 100
    df["log_return_pct"]   = df["log_return"]  * 100

    print(f"對齊後共 {len(df)} 條記錄")
    return df


def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute three correlations with Winsorization."""
    fr  = winsorize(df["funding_rate_pct"], WINSORIZE_LO, WINSORIZE_HI)
    ret = winsorize(df["log_return_pct"],   WINSORIZE_LO, WINSORIZE_HI)

    # 1. Concurrent
    r1, p1 = stats.pearsonr(fr, ret)

    # 2. Predictive (no look-ahead): funding[t] vs return[t+1]
    fr_pred  = fr.iloc[:-1].values
    ret_next = ret.iloc[1:].values
    r2, p2   = stats.pearsonr(fr_pred, ret_next)

    # 3. Cumulative
    cum_fr  = fr.cumsum().values
    cum_ret = ret.cumsum().values
    r3, p3  = stats.pearsonr(cum_fr, cum_ret)

    results = {
        "concurrent_corr":  round(r1, 4),
        "concurrent_pval":  round(p1, 4),
        "predictive_corr":  round(r2, 4),
        "predictive_pval":  round(p2, 4),
        "cumulative_corr":  round(r3, 4),
        "cumulative_pval":  round(p3, 4),
        "n_samples":        len(df),
        "n_pred_samples":   len(fr_pred),
    }

    # Also store winsorized series for plotting
    results["_fr"]      = fr.values
    results["_ret"]     = ret.values
    results["_fr_pred"] = fr_pred
    results["_ret_next"]= ret_next
    results["_cum_fr"]  = cum_fr
    results["_cum_ret"] = cum_ret
    results["_hours"]   = df["hour"].values

    return results


def plot_charts(df: pd.DataFrame, res: dict, output_path: str):
    """Generate 3-panel chart and save to PNG."""
    fr      = res["_fr"]
    ret     = res["_ret"]
    fr_pred = res["_fr_pred"]
    ret_next= res["_ret_next"]
    cum_fr  = res["_cum_fr"]
    cum_ret = res["_cum_ret"]
    hours   = pd.to_datetime(res["_hours"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"HYPE — Funding Rate vs Log Return（過去 {MONTHS_BACK} 個月，Winsorized 1–99%）",
        fontsize=13, fontweight="bold", y=1.02
    )

    # ── Chart 1: Time series ──────────────────────────────────────────
    ax1 = axes[0]
    ax1_r = ax1.twinx()
    ax1.plot(hours, fr,  color="#378ADD", linewidth=0.7, alpha=0.85, label="Funding rate %")
    ax1_r.plot(hours, ret, color="#639922", linewidth=0.6, alpha=0.7, linestyle="--", label="Log return %")
    ax1.set_ylabel("Funding rate (hourly %)", color="#378ADD", fontsize=9)
    ax1_r.set_ylabel("Log return (hourly %)", color="#639922", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax1.set_title("時序圖", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#378ADD", labelsize=8)
    ax1_r.tick_params(axis="y", labelcolor="#639922", labelsize=8)
    lines1 = [plt.Line2D([0],[0],color="#378ADD",lw=1.5),
               plt.Line2D([0],[0],color="#639922",lw=1.5,linestyle="--")]
    ax1.legend(lines1, ["Funding rate %","Log return %"], fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.2)

    # ── Chart 2: Scatter funding[t] vs return[t+1] ────────────────────
    ax2 = axes[1]
    ax2.scatter(fr_pred, ret_next, alpha=0.15, s=6, color="#378ADD")
    m, b, *_ = stats.linregress(fr_pred, ret_next)
    x_line = np.linspace(fr_pred.min(), fr_pred.max(), 100)
    ax2.plot(x_line, m*x_line+b, color="#D85A30", linewidth=1.5, label=f"Trend line")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax2.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_xlabel("Funding rate [t] (%)", fontsize=9)
    ax2.set_ylabel("Log return [t+1] (%)", fontsize=9)
    ax2.set_title(
        f"預測力 scatter\nCorrelation = {res['predictive_corr']:.4f}  (p={res['predictive_pval']:.3f})",
        fontsize=11
    )
    ax2.tick_params(labelsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=8)

    # ── Chart 3: Cumulative ───────────────────────────────────────────
    ax3 = axes[2]
    ax3_r = ax3.twinx()
    ax3.plot(hours, cum_fr,  color="#378ADD", linewidth=1.2, label="Cum funding %")
    ax3_r.plot(hours, cum_ret, color="#D85A30", linewidth=1.2, linestyle="--", label="Cum log return %")
    ax3.set_ylabel("Cumulative funding (%)", color="#378ADD", fontsize=9)
    ax3_r.set_ylabel("Cumulative log return (%)", color="#D85A30", fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax3.set_title(
        f"累積比較\nCorrelation = {res['cumulative_corr']:.4f}  (p={res['cumulative_pval']:.3f})",
        fontsize=11
    )
    ax3.tick_params(axis="y", labelcolor="#378ADD", labelsize=8)
    ax3_r.tick_params(axis="y", labelcolor="#D85A30", labelsize=8)
    lines3 = [plt.Line2D([0],[0],color="#378ADD",lw=1.5),
               plt.Line2D([0],[0],color="#D85A30",lw=1.5,linestyle="--")]
    ax3.legend(lines3, ["Cum funding","Cum log ret"], fontsize=8, loc="upper left")
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"圖表已存：{output_path}")
    plt.show()


def main():
    print("=" * 55)
    print(f"  HYPE Funding Rate vs Log Return — 過去 {MONTHS_BACK} 個月")
    print("=" * 55)

    start_ms = int((datetime.now(timezone.utc) - timedelta(days=MONTHS_BACK*30)).timestamp() * 1000)

    # 1. Download
    funding_df = fetch_funding_history(COIN, start_ms)
    candle_df  = fetch_candles(COIN, start_ms)

    # 2. Align
    df = align_data(funding_df, candle_df)

    # 3. Save aligned CSV
    csv_path = "hype_aligned.csv"
    df[["hour","close","log_return_pct","funding_rate_pct","premium"]].to_csv(csv_path, index=False)
    print(f"對齊數據已存：{csv_path}")

    # 4. Correlations
    print("\n計算 correlation...")
    res = compute_correlations(df)

    # 5. Print results
    print("\n" + "─" * 45)
    print(f"  同期 correlation  (fr[t] vs ret[t])   : {res['concurrent_corr']:+.4f}  (p={res['concurrent_pval']:.4f})")
    print(f"  預測 correlation  (fr[t] vs ret[t+1]) : {res['predictive_corr']:+.4f}  (p={res['predictive_pval']:.4f})  ← 最重要")
    print(f"  累積 correlation  (cum_fr vs cum_ret)  : {res['cumulative_corr']:+.4f}  (p={res['cumulative_pval']:.4f})")
    print(f"  樣本數：{res['n_samples']} 小時（預測用 {res['n_pred_samples']}）")
    print("─" * 45)

    # Interpretation
    r = res["predictive_corr"]
    p = res["predictive_pval"]
    sig = "顯著" if p < 0.05 else "不顯著（p > 0.05）"
    if abs(r) < 0.05:
        interp = "funding rate 對下一小時 return 基本上冇預測力"
    elif r < 0:
        interp = f"負相關 ({r:+.4f})，高 funding 後有 mean reversion 傾向"
    else:
        interp = f"正相關 ({r:+.4f})，高 funding 後 momentum 持續"
    print(f"\n  解讀：{interp}（統計上{sig}）")

    # 6. Save correlation CSV
    corr_df = pd.DataFrame([{
        "metric":           "concurrent",
        "description":      "fr[t] vs ret[t]",
        "correlation":      res["concurrent_corr"],
        "p_value":          res["concurrent_pval"],
        "significant_p05":  res["concurrent_pval"] < 0.05,
    },{
        "metric":           "predictive",
        "description":      "fr[t] vs ret[t+1]  (no look-ahead)",
        "correlation":      res["predictive_corr"],
        "p_value":          res["predictive_pval"],
        "significant_p05":  res["predictive_pval"] < 0.05,
    },{
        "metric":           "cumulative",
        "description":      "cumulative fr vs cumulative ret",
        "correlation":      res["cumulative_corr"],
        "p_value":          res["cumulative_pval"],
        "significant_p05":  res["cumulative_pval"] < 0.05,
    }])
    corr_df.to_csv("hype_correlation.csv", index=False)
    print("\n  Correlation 結果已存：hype_correlation.csv")

    # 7. Plot
    print("\n生成圖表...")
    plot_charts(df, res, "hype_charts.png")

    print("\n完成！輸出檔案：")
    print("  hype_aligned.csv      — 對齊後完整數據")
    print("  hype_correlation.csv  — Correlation 結果")
    print("  hype_charts.png       — 三張分析圖")


if __name__ == "__main__":
    main()
