import argparse
import os
import time
from pathlib import Path

import pandas as pd

try:
    from jqdatasdk import auth, get_all_securities, get_extras, get_industry, get_price, get_valuation
except ImportError:  # pragma: no cover
    auth = None
    get_all_securities = None
    get_extras = None
    get_industry = None
    get_price = None
    get_valuation = None


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch A-share data from JQData and export csv files for backtest_real_data.py.")
    parser.add_argument("--username", help="JoinQuant account username. If omitted, read from JQDATA_USERNAME.")
    parser.add_argument("--password", help="JoinQuant account password. If omitted, read from JQDATA_PASSWORD.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD.")
    parser.add_argument("--output-dir", default="datas/jqdata_export", help="Directory for exported csv files.")
    parser.add_argument("--tickers", help="Optional comma-separated JoinQuant tickers, for example 000001.XSHE,600000.XSHG.")
    parser.add_argument("--min-list-days", type=int, default=120, help="Exclude newly listed stocks younger than this many days at end date.")
    parser.add_argument("--limit", type=int, help="Optional max number of stocks after filtering, useful for quick tests.")
    parser.add_argument("--fq", choices=["pre", "post", "none"], default="pre", help="Price adjustment mode. `pre` is forward-adjusted.")
    parser.add_argument("--skip-st", action="store_true", help="Drop likely ST stocks based on display name.")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for JQData multi-security requests.")
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


def ensure_sdk():
    if auth is None or get_price is None or get_extras is None or get_valuation is None:
        raise ImportError("jqdatasdk is not installed. Run `pip install jqdatasdk` or add it to your environment.")


def normalize_fq(fq):
    return None if fq == "none" else fq


def retry_call(fn, retries=3, sleep_seconds=0.8, **kwargs):
    last_error = None
    for attempt in range(retries):
        try:
            return fn(**kwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
            time.sleep(sleep_seconds * (attempt + 1))
    raise last_error


def login(username, password):
    ensure_sdk()
    if not username or not password:
        raise ValueError("JoinQuant username and password are required. Pass `--username/--password` or set JQDATA_USERNAME/JQDATA_PASSWORD.")
    auth(username, password)


def fetch_universe(args):
    securities = retry_call(get_all_securities, types=["stock"], date=args.end)
    securities = securities.reset_index().rename(columns={"index": "ticker"})
    securities["ticker"] = securities["ticker"].astype(str)
    securities["start_date"] = pd.to_datetime(securities["start_date"], errors="coerce")
    securities["end_date"] = pd.to_datetime(securities["end_date"], errors="coerce")

    end_date = pd.to_datetime(args.end)
    if args.min_list_days:
        securities = securities[(end_date - securities["start_date"]).dt.days >= args.min_list_days].copy()

    if args.tickers:
        selected = {ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()}
        securities = securities[securities["ticker"].isin(selected)].copy()

    if args.skip_st:
        securities = securities[~securities["display_name"].astype(str).str.contains("ST", na=False)].copy()

    securities = securities.sort_values("ticker").reset_index(drop=True)
    if args.limit:
        securities = securities.head(args.limit).copy()
    return securities


def flatten_industry_record(record):
    if not isinstance(record, dict):
        return {"sector": None, "industry": None, "subindustry": None}
    return {
        "sector": (record.get("sw_l1") or {}).get("industry_name"),
        "industry": (record.get("sw_l2") or {}).get("industry_name"),
        "subindustry": (record.get("sw_l3") or {}).get("industry_name"),
    }


def fetch_industry_snapshot(tickers, date):
    rows = []
    batch_size = 200
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start : start + batch_size]
        try:
            industry_map = retry_call(get_industry, securities=batch, date=date)
        except Exception:  # pragma: no cover
            industry_map = {}
        for ticker in batch:
            row = {"ticker": ticker}
            row.update(flatten_industry_record(industry_map.get(ticker)))
            rows.append(row)
    return pd.DataFrame(rows)


def fetch_price_panel(tickers, start, end, fq):
    return retry_call(
        get_price,
        security=tickers,
        start_date=start,
        end_date=end,
        frequency="daily",
        fields=["open", "high", "low", "close", "volume", "money", "paused", "high_limit", "low_limit"],
        skip_paused=False,
        fq=fq,
        fill_paused=False,
        panel=False,
    )


def fetch_valuation_panel(tickers, start, end):
    last_error = None
    for attempt in range(3):
        try:
            return get_valuation(
                tickers,
                start_date=start,
                end_date=end,
                fields=["day", "market_cap", "circulating_market_cap", "turnover_ratio"],
            )
        except Exception as exc:  # pragma: no cover
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    raise last_error


def fetch_is_st_matrix(tickers, start, end):
    return retry_call(get_extras, info="is_st", security_list=tickers, start_date=start, end_date=end)


def fetch_prices(tickers, start, end, fq, batch_size):
    price_parts = []
    failed = {}
    is_st_matrix = fetch_is_st_matrix(tickers, start, end)
    for start_idx in range(0, len(tickers), batch_size):
        batch = tickers[start_idx : start_idx + batch_size]
        try:
            frame = fetch_price_panel(batch, start, end, fq)
        except Exception as exc:  # pragma: no cover
            for ticker in batch:
                failed[ticker] = f"price failed: {exc}"
            continue
        if frame is None or frame.empty:
            for ticker in batch:
                failed[ticker] = "empty price result"
            continue
        try:
            valuation = fetch_valuation_panel(batch, start, end)
        except Exception as exc:  # pragma: no cover
            for ticker in batch:
                failed[ticker] = f"valuation failed: {exc}"
            continue
        frame = frame.rename(columns={"time": "date", "code": "ticker", "money": "amount"})
        frame["date"] = pd.to_datetime(frame["date"])
        frame["ticker"] = frame["ticker"].astype(str)

        valuation = valuation.rename(columns={"day": "date", "code": "ticker"})
        valuation["date"] = pd.to_datetime(valuation["date"])
        valuation["ticker"] = valuation["ticker"].astype(str)

        frame = frame.merge(valuation, on=["date", "ticker"], how="left")
        st_part = is_st_matrix.reindex(columns=batch).copy()
        st_part.index = pd.to_datetime(st_part.index)
        st_part = (
            st_part.rename_axis("date")
            .reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="is_st")
        )
        st_part["ticker"] = st_part["ticker"].astype(str)
        frame = frame.merge(st_part, on=["date", "ticker"], how="left")
        frame["is_st"] = frame["is_st"].fillna(False)

        frame["cap"] = frame["market_cap"] * 100000000.0
        frame["circulating_cap"] = frame["circulating_market_cap"] * 100000000.0
        price_parts.append(
            frame[
                [
                    "date",
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "paused",
                    "high_limit",
                    "low_limit",
                    "is_st",
                    "cap",
                    "circulating_cap",
                    "turnover_ratio",
                ]
            ]
        )

    if not price_parts:
        details = ", ".join(f"{ticker}: {reason}" for ticker, reason in failed.items())
        raise ValueError(f"No daily data returned from JQData. Details: {details}")

    prices = pd.concat(price_parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    return prices, failed


def build_meta(universe, industry):
    meta = universe.merge(industry, on="ticker", how="left")
    meta["sector"] = meta["sector"].fillna("unknown")
    meta["industry"] = meta["industry"].fillna(meta["sector"])
    meta["subindustry"] = meta["subindustry"].fillna(meta["industry"])
    meta["exchange"] = meta["ticker"].str[-4:].map({"XSHE": "SZSE", "XSHG": "SSE"})
    meta["market"] = meta["exchange"].map({"SZSE": "A-share", "SSE": "A-share"}).fillna("A-share")
    meta["list_status"] = "L"
    meta["symbol"] = meta["ticker"].str[:6]
    meta["name"] = meta["display_name"]
    meta["list_date"] = meta["start_date"].dt.strftime("%Y%m%d")
    meta["delist_date"] = meta["end_date"].dt.strftime("%Y%m%d")
    meta = meta[
        [
            "ticker",
            "symbol",
            "name",
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
    return meta


def main():
    args = parse_args()
    load_local_env()
    username = args.username or os.getenv("JQDATA_USERNAME")
    password = args.password or os.getenv("JQDATA_PASSWORD")
    login(username, password)

    universe = fetch_universe(args)
    if universe.empty:
        raise ValueError("No stocks matched the requested filters.")

    tickers = universe["ticker"].tolist()
    industry = fetch_industry_snapshot(tickers, args.end)
    prices, failed = fetch_prices(tickers, args.start, args.end, normalize_fq(args.fq), args.batch_size)
    meta = build_meta(universe, industry)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prices_path = output_dir / "prices.csv"
    meta_path = output_dir / "meta.csv"
    prices.to_csv(prices_path, index=False)
    meta.to_csv(meta_path, index=False)

    print("JQData export completed")
    print(f"Prices csv: {prices_path}")
    print(f"Meta csv:   {meta_path}")
    print(f"Stocks:     {meta['ticker'].nunique()}")
    print(f"Rows:       {len(prices)}")
    print(f"Date range: {prices['date'].min().date()} -> {prices['date'].max().date()}")
    if failed:
        print("Warning: failed tickers from JQData")
        for ticker, reason in sorted(failed.items()):
            print(f"  {ticker}: {reason}")


if __name__ == "__main__":
    main()
