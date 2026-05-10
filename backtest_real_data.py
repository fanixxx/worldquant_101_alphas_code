import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

from alphas.alpha101 import Alphas


PRICE_FIELDS = ["Open", "High", "Low", "Close", "Volume"]
GROUP_LEVELS = ["sector", "industry", "subindustry"]


@dataclass
class BacktestResult:
    ic_mean: float
    ic_ir: float
    annual_return_gross: float
    annual_return_net: float
    annual_vol: float
    sharpe_gross: float
    sharpe_net: float
    max_drawdown_gross: float
    max_drawdown_net: float
    win_rate_gross: float
    win_rate_net: float
    average_turnover: float
    cost_bps: float


def parse_args():
    parser = argparse.ArgumentParser(description="Use real stock data to backtest Alpha101 factors.")
    parser.add_argument("--alpha", default="alpha_001", help="Alpha method name, for example alpha_001.")
    parser.add_argument("--alphas", help="Optional comma-separated alpha names to run in batch. Overrides `--alpha`.")
    parser.add_argument("--tickers", default="AAPL,MSFT,GOOG,AMZN,META,NVDA,TSLA,JPM,JNJ,XOM", help="Comma-separated ticker list.")
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date.")
    parser.add_argument("--end", default="2025-01-01", help="Backtest end date.")
    parser.add_argument("--source", choices=["yfinance", "csv"], default="yfinance", help="Data source.")
    parser.add_argument("--csv", help="CSV path when source=csv. Expect columns: date,ticker,open,high,low,close,volume.")
    parser.add_argument("--meta-csv", help="Optional metadata CSV with columns like ticker,sector,industry,subindustry,cap.")
    parser.add_argument("--long-quantile", type=float, default=0.8, help="Long when factor rank >= this quantile.")
    parser.add_argument("--short-quantile", type=float, default=0.2, help="Short when factor rank <= this quantile.")
    parser.add_argument("--n-quantiles", type=int, default=5, help="Number of quantile buckets for layer analysis.")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="One-way transaction cost in basis points.")
    parser.add_argument("--neutralize", choices=["none", "sector", "industry", "subindustry"], default="none", help="Optional cross-sectional group neutralization before backtest.")
    parser.add_argument("--min-amount", type=float, default=2e7, help="Minimum same-day traded amount to allow a stock into the signal, in currency units.")
    parser.add_argument("--min-price", type=float, default=2.0, help="Minimum same-day close price to allow a stock into the signal.")
    parser.add_argument("--exclude-flat-bars", action="store_true", help="Exclude days where high equals low, a rough proxy for one-price or non-tradable bars.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write csv summaries and plots.")
    return parser.parse_args()


def download_from_yfinance(tickers, start, end):
    if yf is None:
        raise ImportError("yfinance is not installed. Run `pip install -r requirements.txt` first.")
    frames = {field.lower(): [] for field in PRICE_FIELDS}
    failed = {}

    for ticker in tickers:
        last_error = None
        history = pd.DataFrame()
        for attempt in range(3):
            try:
                history = yf.Ticker(ticker).history(
                    start=start,
                    end=end,
                    auto_adjust=False,
                    actions=False,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                history = pd.DataFrame()
            if not history.empty:
                break
            time.sleep(1.5 * (attempt + 1))

        if history.empty:
            failed[ticker] = str(last_error) if last_error is not None else "empty result"
            continue

        history.index = pd.to_datetime(history.index).tz_localize(None)
        for field in PRICE_FIELDS:
            field_df = history[[field]].rename(columns={field: ticker})
            frames[field.lower()].append(field_df)

    if not any(frames.values()):
        details = ", ".join(f"{ticker}: {reason}" for ticker, reason in failed.items())
        raise ValueError(f"No data returned from yfinance. Details: {details}")

    data = {}
    for field, items in frames.items():
        if not items:
            raise ValueError(f"No valid `{field}` data returned from yfinance.")
        data[field] = pd.concat(items, axis=1).sort_index()

    if failed:
        print("Warning: failed tickers from yfinance")
        for ticker, reason in failed.items():
            print(f"  {ticker}: {reason}")
    return data


def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    data = {}
    for field in ["open", "high", "low", "close", "volume"]:
        pivot = df.pivot(index="date", columns="ticker", values=field).sort_index()
        data[field] = pivot
    for field in ["cap", "amount", "dollar_volume", "paused", "high_limit", "low_limit", "is_st", "circulating_cap", "turnover_ratio"]:
        if field in df.columns:
            data[field] = df.pivot(index="date", columns="ticker", values=field).sort_index()
    return data


def load_metadata_csv(meta_csv_path):
    meta = pd.read_csv(meta_csv_path)
    if "ticker" not in meta.columns:
        raise ValueError("Metadata CSV must contain a `ticker` column.")
    meta["ticker"] = meta["ticker"].astype(str)
    return meta.drop_duplicates(subset=["ticker"]).set_index("ticker")


def build_alpha_input(price_data, metadata=None):
    close = price_data["close"]
    open_ = price_data["open"]
    high = price_data["high"]
    low = price_data["low"]
    volume = price_data["volume"]
    vwap = (open_ + high + low + close) / 4.0
    dollar_volume = price_data["dollar_volume"] if "dollar_volume" in price_data else None
    if dollar_volume is None and "amount" in price_data:
        dollar_volume = price_data["amount"]
    if dollar_volume is None:
        dollar_volume = vwap * volume
    if metadata is None:
        group = pd.Series({ticker: "all" for ticker in close.columns})
        sector = group
        industry = group
        subindustry = group
        cap = price_data["cap"] if "cap" in price_data else None
    else:
        metadata = metadata.reindex(close.columns)
        sector = metadata["sector"] if "sector" in metadata.columns else pd.Series("all", index=close.columns)
        industry = metadata["industry"] if "industry" in metadata.columns else pd.Series("all", index=close.columns)
        subindustry = metadata["subindustry"] if "subindustry" in metadata.columns else pd.Series("all", index=close.columns)
        if "cap" in metadata.columns:
            if metadata["cap"].notna().all():
                cap = pd.DataFrame(
                    np.tile(metadata["cap"].to_numpy(), (len(close.index), 1)),
                    index=close.index,
                    columns=close.columns,
                )
            else:
                cap = None
        elif "shares_outstanding" in metadata.columns:
            cap = close.mul(metadata["shares_outstanding"], axis=1) if metadata["shares_outstanding"].notna().all() else None
        else:
            cap = price_data["cap"] if "cap" in price_data else None
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "dollar_volume": dollar_volume,
        "vwap": vwap,
        "cap": cap,
        "sector": sector,
        "industry": industry,
        "subindustry": subindustry,
    }


def compute_factor(data, alpha_name):
    alphas = Alphas(data)
    if not hasattr(alphas, alpha_name):
        raise ValueError(f"Alpha `{alpha_name}` does not exist.")
    if alpha_name == "alpha_056" and data.get("cap") is None:
        raise ValueError("alpha_056 requires real market cap data. Provide `--meta-csv` with a `cap` column.")
    factor = getattr(alphas, alpha_name)()
    return factor.replace([np.inf, -np.inf], np.nan)


def parse_alpha_names(args):
    if args.alphas:
        return [item.strip() for item in args.alphas.split(",") if item.strip()]
    return [args.alpha]


def build_tradeable_mask(price_data, min_amount, min_price, exclude_flat_bars):
    close = price_data["close"]
    open_ = price_data["open"]
    volume = price_data["volume"]
    amount = price_data.get("amount")
    if amount is None and "dollar_volume" in price_data:
        amount = price_data["dollar_volume"]
    if amount is None:
        amount = close * volume

    mask = close.notna() & open_.notna() & volume.notna()
    mask &= volume > 0
    mask &= close >= min_price
    mask &= amount >= min_amount
    if "paused" in price_data:
        mask &= price_data["paused"].fillna(1.0) == 0
    if "is_st" in price_data:
        st_frame = price_data["is_st"].fillna(True).astype(bool)
        mask &= ~st_frame

    if exclude_flat_bars:
        mask &= (price_data["high"] > price_data["low"])

    buyable = mask.copy()
    shortable = mask.copy()
    if "high_limit" in price_data:
        buyable &= open_ < price_data["high_limit"]
    if "low_limit" in price_data:
        shortable &= open_ > price_data["low_limit"]
    return {
        "signal_mask": mask,
        "buyable_mask": buyable,
        "shortable_mask": shortable,
    }


def neutralize_factor(factor, group_labels):
    if group_labels is None:
        return factor
    labels = pd.Series(group_labels).reindex(factor.columns)
    out = factor.copy()
    for group_name in labels.dropna().unique():
        cols = labels[labels == group_name].index.tolist()
        if cols:
            out[cols] = out[cols].sub(out[cols].mean(axis=1), axis=0)
    return out


def to_quantile_labels(factor, n_quantiles):
    ranks = factor.rank(axis=1, pct=True, method="first")
    quantiles = np.ceil(ranks * n_quantiles).clip(1, n_quantiles)
    return quantiles.where(factor.notna())


def compute_positions(factor, long_quantile, short_quantile):
    ranks = factor.rank(axis=1, pct=True)
    long_mask = ranks >= long_quantile
    short_mask = ranks <= short_quantile

    long_weight = long_mask.div(long_mask.sum(axis=1), axis=0).fillna(0.0)
    short_weight = short_mask.div(short_mask.sum(axis=1), axis=0).fillna(0.0)
    return long_weight - short_weight


def normalize_side(weights):
    denom = weights.sum(axis=1).replace(0.0, np.nan)
    return weights.div(denom, axis=0).fillna(0.0)


def apply_execution_constraints(raw_positions, next_buyable, next_shortable):
    long_weight = raw_positions.clip(lower=0.0).where(next_buyable, 0.0)
    short_weight = (-raw_positions.clip(upper=0.0)).where(next_shortable, 0.0)
    long_weight = normalize_side(long_weight)
    short_weight = normalize_side(short_weight)
    return long_weight - short_weight


def compute_quantile_returns(factor, next_ret, n_quantiles):
    quantile_labels = to_quantile_labels(factor, n_quantiles)
    quantile_returns = {}
    for q in range(1, n_quantiles + 1):
        mask = quantile_labels == q
        weight = mask.div(mask.sum(axis=1), axis=0)
        quantile_returns[f"Q{q}"] = (weight * next_ret).sum(axis=1, min_count=1)
    return pd.DataFrame(quantile_returns).dropna(how="all")


def compute_turnover(positions):
    shifted = positions.shift(1).fillna(0.0)
    return positions.fillna(0.0).sub(shifted).abs().sum(axis=1)


def annualized_return(daily_returns):
    return float(daily_returns.mean() * 252)


def annualized_vol(daily_returns):
    return float(daily_returns.std(ddof=0) * np.sqrt(252))


def max_drawdown(daily_returns):
    nav = (1 + daily_returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min())


def sharpe_ratio(daily_returns):
    vol = annualized_vol(daily_returns)
    return float(annualized_return(daily_returns) / vol) if vol > 0 else np.nan


def performance_stats(gross_returns, net_returns, turnover, cost_bps):
    gross_returns = gross_returns.dropna()
    net_returns = net_returns.dropna()
    if gross_returns.empty or net_returns.empty:
        raise ValueError("No valid portfolio returns produced. Expand ticker universe or backtest window.")
    return BacktestResult(
        ic_mean=np.nan,
        ic_ir=np.nan,
        annual_return_gross=annualized_return(gross_returns),
        annual_return_net=annualized_return(net_returns),
        annual_vol=annualized_vol(net_returns),
        sharpe_gross=sharpe_ratio(gross_returns),
        sharpe_net=sharpe_ratio(net_returns),
        max_drawdown_gross=max_drawdown(gross_returns),
        max_drawdown_net=max_drawdown(net_returns),
        win_rate_gross=float((gross_returns > 0).mean()),
        win_rate_net=float((net_returns > 0).mean()),
        average_turnover=float(turnover.mean()),
        cost_bps=float(cost_bps),
    )


def evaluate(factor, close, tradeable_mask, long_quantile, short_quantile, n_quantiles, cost_bps):
    next_ret = close.pct_change(fill_method=None).shift(-1)
    signal_mask = tradeable_mask["signal_mask"]
    next_buyable = tradeable_mask["buyable_mask"].shift(-1)
    next_shortable = tradeable_mask["shortable_mask"].shift(-1)
    next_buyable = next_buyable.where(next_buyable.notna(), False).astype(bool)
    next_shortable = next_shortable.where(next_shortable.notna(), False).astype(bool)
    next_tradeable = next_buyable | next_shortable
    factor = factor.where(signal_mask)
    next_ret = next_ret.where(next_tradeable)
    ic_series = factor.corrwith(next_ret, axis=1, method="spearman").dropna()

    raw_positions = compute_positions(factor, long_quantile, short_quantile)
    positions = apply_execution_constraints(raw_positions, next_buyable, next_shortable)
    gross_returns = (positions * next_ret).sum(axis=1).dropna()
    turnover = compute_turnover(positions).reindex(gross_returns.index).fillna(0.0)
    cost_rate = cost_bps / 10000.0
    net_returns = gross_returns - turnover * cost_rate

    quantile_returns = compute_quantile_returns(factor, next_ret, n_quantiles)
    quantile_nav = (1 + quantile_returns.fillna(0.0)).cumprod()

    result = performance_stats(gross_returns, net_returns, turnover, cost_bps)
    ic_mean = float(ic_series.mean()) if not ic_series.empty else np.nan
    ic_std = float(ic_series.std(ddof=0)) if not ic_series.empty else np.nan
    result.ic_mean = ic_mean
    result.ic_ir = float(ic_mean / ic_std) if np.isfinite(ic_std) and ic_std > 0 else np.nan

    summary_frame = pd.DataFrame(
        {
            "gross_return": gross_returns,
            "net_return": net_returns,
            "turnover": turnover,
            "ic": ic_series.reindex(gross_returns.index),
        }
    )
    return result, summary_frame, quantile_returns, quantile_nav, positions


def print_result(result, summary_frame, quantile_returns):
    print("Backtest summary")
    print(f"IC mean:           {result.ic_mean:.6f}")
    print(f"IC IR:             {result.ic_ir:.6f}")
    print(f"Annual return(g):  {result.annual_return_gross:.2%}")
    print(f"Annual return(n):  {result.annual_return_net:.2%}")
    print(f"Annual vol(n):     {result.annual_vol:.2%}")
    print(f"Sharpe(g):         {result.sharpe_gross:.4f}")
    print(f"Sharpe(n):         {result.sharpe_net:.4f}")
    print(f"Max drawdown(g):   {result.max_drawdown_gross:.2%}")
    print(f"Max drawdown(n):   {result.max_drawdown_net:.2%}")
    print(f"Win rate(g):       {result.win_rate_gross:.2%}")
    print(f"Win rate(n):       {result.win_rate_net:.2%}")
    print(f"Avg turnover:      {result.average_turnover:.4f}")
    print(f"Cost bps:          {result.cost_bps:.2f}")
    print(f"Days used:         {len(summary_frame)}")
    print(f"IC samples:        {summary_frame['ic'].notna().sum()}")
    print(f"Quantile columns:  {', '.join(quantile_returns.columns)}")


def save_plot(series_or_frame, title, ylabel, output_path):
    if plt is None:
        return
    ax = series_or_frame.plot(figsize=(12, 6), title=title, linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=150)
    plt.close(ax.figure)


def save_outputs(args, alpha_name, factor, result, summary_frame, quantile_returns, quantile_nav, positions):
    output_dir = Path(args.output_dir) / alpha_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_row = pd.DataFrame([result.__dict__])
    summary_row["alpha"] = alpha_name
    summary_row["neutralize"] = args.neutralize
    summary_row["n_quantiles"] = args.n_quantiles
    summary_row["min_amount"] = args.min_amount
    summary_row["min_price"] = args.min_price
    summary_row["exclude_flat_bars"] = args.exclude_flat_bars

    summary_row.to_csv(output_dir / "summary.csv", index=False)
    summary_frame.to_csv(output_dir / "daily_summary.csv")
    quantile_returns.to_csv(output_dir / "quantile_returns.csv")
    quantile_nav.to_csv(output_dir / "quantile_nav.csv")
    factor.to_csv(output_dir / "factor_values.csv")
    positions.to_csv(output_dir / "positions.csv")

    net_nav = (1 + summary_frame["net_return"].fillna(0.0)).cumprod()
    gross_nav = (1 + summary_frame["gross_return"].fillna(0.0)).cumprod()
    nav_frame = pd.DataFrame({"gross_nav": gross_nav, "net_nav": net_nav})
    nav_frame.to_csv(output_dir / "portfolio_nav.csv")

    if plt is None:
        print("Warning: matplotlib is not installed, so png plots were not generated.")
        return

    save_plot(nav_frame, f"{alpha_name} Portfolio NAV", "NAV", output_dir / "portfolio_nav.png")
    if summary_frame["ic"].dropna().empty:
        print("Warning: no valid IC series, skipped ic_cumsum.png.")
    else:
        save_plot(summary_frame["ic"].dropna().cumsum(), f"{alpha_name} IC Cumulative Sum", "IC CumSum", output_dir / "ic_cumsum.png")
    save_plot(quantile_nav, f"{alpha_name} Quantile NAV", "NAV", output_dir / "quantile_nav.png")


def main():
    args = parse_args()
    alpha_names = parse_alpha_names(args)
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    if len(tickers) < 5 and args.source == "yfinance":
        raise ValueError("Use at least 5 tickers. Cross-sectional alphas need a stock universe, not a single stock.")
    if not (0 < args.short_quantile < args.long_quantile < 1):
        raise ValueError("Quantile thresholds must satisfy 0 < short < long < 1.")
    if args.n_quantiles < 2:
        raise ValueError("`--n-quantiles` must be at least 2.")

    if args.source == "yfinance":
        price_data = download_from_yfinance(tickers, args.start, args.end)
    else:
        if not args.csv:
            raise ValueError("`--csv` is required when source=csv.")
        price_data = load_from_csv(args.csv)

    metadata = load_metadata_csv(args.meta_csv) if args.meta_csv else None
    alpha_input = build_alpha_input(price_data, metadata=metadata)
    tradeable_mask = build_tradeable_mask(
        price_data,
        min_amount=args.min_amount,
        min_price=args.min_price,
        exclude_flat_bars=args.exclude_flat_bars,
    )

    batch_rows = []
    batch_errors = []
    for alpha_name in alpha_names:
        try:
            factor = compute_factor(alpha_input, alpha_name)

            if args.neutralize != "none":
                if metadata is None:
                    raise ValueError("Group neutralization requires `--meta-csv`.")
                factor = neutralize_factor(factor, alpha_input[args.neutralize])

            result, summary_frame, quantile_returns, quantile_nav, positions = evaluate(
                factor,
                alpha_input["close"],
                tradeable_mask,
                args.long_quantile,
                args.short_quantile,
                args.n_quantiles,
                args.cost_bps,
            )
            print_result(result, summary_frame, quantile_returns)
            save_outputs(args, alpha_name, factor, result, summary_frame, quantile_returns, quantile_nav, positions)

            row = dict(result.__dict__)
            row["alpha"] = alpha_name
            batch_rows.append(row)
        except Exception as exc:
            print(f"Alpha failed: {alpha_name}: {exc}")
            batch_errors.append({"alpha": alpha_name, "error": str(exc)})

    if len(batch_rows) > 1:
        batch_summary = pd.DataFrame(batch_rows).sort_values(["ic_mean", "sharpe_net"], ascending=[False, False])
        batch_summary.to_csv(Path(args.output_dir) / "batch_summary.csv", index=False)
        print(f"Batch summary written to {Path(args.output_dir) / 'batch_summary.csv'}")
    if batch_errors:
        pd.DataFrame(batch_errors).to_csv(Path(args.output_dir) / "batch_errors.csv", index=False)
        print(f"Batch errors written to {Path(args.output_dir) / 'batch_errors.csv'}")


if __name__ == "__main__":
    main()
