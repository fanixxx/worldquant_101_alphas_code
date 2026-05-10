import argparse
import os
import time
from pathlib import Path

import pandas as pd

try:
    import tushare as ts
except ImportError:  # pragma: no cover
    ts = None


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch A-share data from Tushare Pro and export csv files for backtest_real_data.py.")
    parser.add_argument("--token", help="Tushare Pro token. If omitted, read from TUSHARE_TOKEN.")
    parser.add_argument("--start", required=True, help="Start date in YYYYMMDD.")
    parser.add_argument("--end", required=True, help="End date in YYYYMMDD.")
    parser.add_argument("--output-dir", default="datas/tushare_export", help="Directory for exported csv files.")
    parser.add_argument("--tickers", help="Optional comma-separated TS codes, for example 000001.SZ,600000.SH.")
    parser.add_argument("--exchange", choices=["SSE", "SZSE", "BSE"], help="Optional exchange filter.")
    parser.add_argument("--market", help="Optional market filter such as 主板/创业板/科创板/北交所.")
    parser.add_argument("--list-status", default="L", help="Stock listing status for stock_basic, default is L.")
    parser.add_argument("--min-list-days", type=int, default=120, help="Exclude newly listed stocks younger than this many days at end date.")
    parser.add_argument("--limit", type=int, help="Optional max number of stocks after filtering, useful for quick tests.")
    parser.add_argument("--adj", choices=["raw", "qfq"], default="qfq", help="Price adjustment mode for exported OHLC.")
    return parser.parse_args()


def load_local_env():
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_pro_client(token):
    if ts is None:
        raise ImportError("tushare is not installed. Run `pip install tushare` or add it to your environment.")
    if not token:
        raise ValueError("Tushare token is required. Pass `--token` or set `TUSHARE_TOKEN`.")
    ts.set_token(token)
    return ts.pro_api(token)


def retry_call(fn, retries=3, sleep_seconds=0.6, **kwargs):
    last_error = None
    for attempt in range(retries):
        try:
            return fn(**kwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
            time.sleep(sleep_seconds * (attempt + 1))
    raise last_error


def fetch_stock_basic(pro, args):
    df = retry_call(
        pro.stock_basic,
        exchange=args.exchange,
        list_status=args.list_status,
        fields="ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date",
    )
    if args.market:
        df = df[df["market"] == args.market].copy()
    if args.tickers:
        selected = {ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()}
        df = df[df["ts_code"].isin(selected)].copy()
    if args.min_list_days:
        end_date = pd.to_datetime(args.end)
        list_date = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
        df = df[(end_date - list_date).dt.days >= args.min_list_days].copy()
    df = df[~df["name"].str.contains("ST", na=False)].copy()
    df = df.sort_values("ts_code").reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).copy()
    return df


def fetch_daily_panels(pro, start, end, universe_codes):
    daily_parts = []
    adj_parts = []
    basic_parts = []

    for ts_code in universe_codes:
        daily = retry_call(
            pro.daily,
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields="ts_code,trade_date,open,high,low,close,vol,amount",
        )
        if daily.empty:
            continue

        adj = retry_call(
            pro.adj_factor,
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields="ts_code,trade_date,adj_factor",
        )

        basic = retry_call(
            pro.daily_basic,
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields="ts_code,trade_date,total_mv",
        )

        daily_parts.append(daily)
        adj_parts.append(adj)
        basic_parts.append(basic)

    if not daily_parts:
        raise ValueError("No daily data fetched from Tushare for the requested range.")

    daily_df = pd.concat(daily_parts, ignore_index=True)
    adj_df = pd.concat(adj_parts, ignore_index=True) if adj_parts else pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
    basic_df = pd.concat(basic_parts, ignore_index=True) if basic_parts else pd.DataFrame(columns=["ts_code", "trade_date", "total_mv"])
    return daily_df, adj_df, basic_df


def apply_adjustment(df, mode):
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    if mode == "raw":
        for col in ["open", "high", "low", "close"]:
            df[f"{col}_adj"] = df[col]
        return df

    if "adj_factor" not in df.columns:
        raise ValueError("adj_factor is required for adjusted prices.")

    latest_factor = df.groupby("ts_code")["adj_factor"].transform("last")
    ratio = df["adj_factor"] / latest_factor
    for col in ["open", "high", "low", "close"]:
        df[f"{col}_adj"] = df[col] * ratio
    return df


def fetch_sw_industry(pro, universe_codes):
    try:
        industry = retry_call(
            pro.index_member_all,
            is_new="Y",
            fields="ts_code,l1_name,l2_name,l3_name",
        )
    except Exception:  # pragma: no cover
        return pd.DataFrame(columns=["ts_code", "l1_name", "l2_name", "l3_name"])
    industry = industry[industry["ts_code"].isin(set(universe_codes))].copy()
    industry = industry.drop_duplicates(subset=["ts_code"])
    return industry


def build_exports(stock_basic, daily_df, adj_df, basic_df, industry_df, output_dir, adj_mode):
    merged = daily_df.merge(adj_df, on=["ts_code", "trade_date"], how="left")
    merged = merged.merge(basic_df, on=["ts_code", "trade_date"], how="left")
    merged = apply_adjustment(merged, adj_mode)

    merged["date"] = pd.to_datetime(merged["trade_date"], format="%Y%m%d")
    merged["ticker"] = merged["ts_code"]
    merged["volume"] = merged["vol"] * 100
    merged["amount"] = merged["amount"] * 1000
    merged["cap"] = merged["total_mv"] * 10000

    prices = merged[
        [
            "date",
            "ticker",
            "open_adj",
            "high_adj",
            "low_adj",
            "close_adj",
            "volume",
            "amount",
            "cap",
            "adj_factor",
        ]
    ].rename(
        columns={
            "open_adj": "open",
            "high_adj": "high",
            "low_adj": "low",
            "close_adj": "close",
        }
    )
    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)

    meta = stock_basic.merge(industry_df, on="ts_code", how="left")
    meta = meta.rename(
        columns={
            "ts_code": "ticker",
            "industry_x": "tushare_industry",
            "l1_name": "sector",
            "l2_name": "industry",
            "l3_name": "subindustry",
        }
    )
    if "industry_y" in meta.columns:
        meta = meta.drop(columns=["industry_y"])
    meta["sector"] = meta["sector"].fillna(meta["market"])
    meta["industry"] = meta["industry"].fillna(meta["tushare_industry"])
    meta["subindustry"] = meta["subindustry"].fillna(meta["industry"])
    meta = meta[
        [
            "ticker",
            "symbol",
            "name",
            "area",
            "market",
            "exchange",
            "list_status",
            "list_date",
            "delist_date",
            "sector",
            "industry",
            "subindustry",
        ]
    ].sort_values("ticker")

    output_dir.mkdir(parents=True, exist_ok=True)
    prices_path = output_dir / "prices.csv"
    meta_path = output_dir / "meta.csv"
    prices.to_csv(prices_path, index=False)
    meta.to_csv(meta_path, index=False)
    return prices_path, meta_path, prices, meta


def main():
    args = parse_args()
    load_local_env()
    token = args.token or os.getenv("TUSHARE_TOKEN")
    pro = get_pro_client(token)

    stock_basic = fetch_stock_basic(pro, args)
    if stock_basic.empty:
        raise ValueError("No stocks matched the requested filters.")

    daily_df, adj_df, basic_df = fetch_daily_panels(pro, args.start, args.end, stock_basic["ts_code"].tolist())
    industry_df = fetch_sw_industry(pro, stock_basic["ts_code"].tolist())

    output_dir = Path(args.output_dir)
    prices_path, meta_path, prices, meta = build_exports(
        stock_basic,
        daily_df,
        adj_df,
        basic_df,
        industry_df,
        output_dir,
        args.adj,
    )

    print("Tushare export completed")
    print(f"Prices csv: {prices_path}")
    print(f"Meta csv:   {meta_path}")
    print(f"Stocks:     {meta['ticker'].nunique()}")
    print(f"Rows:       {len(prices)}")
    print(f"Date range: {prices['date'].min().date()} -> {prices['date'].max().date()}")


if __name__ == "__main__":
    main()
