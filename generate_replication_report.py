import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a replication validation report from exported data and backtest summaries.")
    parser.add_argument("--prices", required=True, help="Path to exported prices.csv.")
    parser.add_argument("--meta", required=True, help="Path to exported meta.csv.")
    parser.add_argument("--batch-summary", required=True, help="Path to batch_summary.csv.")
    parser.add_argument("--batch-errors", help="Optional path to batch_errors.csv.")
    parser.add_argument("--report-path", default="replication_report.md", help="Output markdown report path.")
    parser.add_argument("--title", default="WorldQuant 101 A-share Replication Validation Report", help="Report title.")
    parser.add_argument("--data-source", default="JQData", help="Data source name.")
    parser.add_argument("--notes", default="", help="Extra note appended to the limitations section.")
    return parser.parse_args()


def load_csv(path):
    return pd.read_csv(path)


def fmt_pct(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.2%}"


def fmt_num(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.4f}"


def top_table(df, columns, limit=10):
    subset = df.loc[:, columns].head(limit).copy()
    return subset.to_markdown(index=False)


def build_report(args, prices, meta, summary, errors):
    prices["date"] = pd.to_datetime(prices["date"])
    if "is_st" in prices.columns:
        st_ratio = float(prices["is_st"].fillna(False).astype(bool).mean())
    else:
        st_ratio = float("nan")
    if "paused" in prices.columns:
        paused_ratio = float((prices["paused"].fillna(1.0) != 0).mean())
    else:
        paused_ratio = float("nan")
    if {"open", "high_limit"}.issubset(prices.columns):
        buy_limit_ratio = float((prices["open"] >= prices["high_limit"]).mean())
    else:
        buy_limit_ratio = float("nan")
    if {"open", "low_limit"}.issubset(prices.columns):
        sell_limit_ratio = float((prices["open"] <= prices["low_limit"]).mean())
    else:
        sell_limit_ratio = float("nan")

    positive_ic = summary[summary["ic_mean"] > 0].copy()
    positive_net = summary[summary["annual_return_net"] > 0].copy()
    best_ic = summary.sort_values(["ic_mean", "sharpe_net"], ascending=[False, False]).reset_index(drop=True)
    best_net = summary.sort_values(["annual_return_net", "ic_mean"], ascending=[False, False]).reset_index(drop=True)

    lines = [
        f"# {args.title}",
        "",
        "## Validation Scope",
        f"- Data source: `{args.data_source}`",
        f"- Stock count: `{meta['ticker'].nunique()}`",
        f"- Price rows: `{len(prices)}`",
        f"- Date range: `{prices['date'].min().date()}` to `{prices['date'].max().date()}`",
        f"- Tested alpha count: `{len(summary)}`",
        f"- Failed alpha count: `{0 if errors is None else len(errors)}`",
        "",
        "## Data Diagnostics",
        f"- ST ratio in exported daily rows: `{fmt_pct(st_ratio)}`",
        f"- Paused ratio in exported daily rows: `{fmt_pct(paused_ratio)}`",
        f"- Open at upper limit ratio: `{fmt_pct(buy_limit_ratio)}`",
        f"- Open at lower limit ratio: `{fmt_pct(sell_limit_ratio)}`",
        "",
        "## Headline Result",
        f"- Positive IC alphas: `{len(positive_ic)}/{len(summary)}`",
        f"- Positive net return alphas: `{len(positive_net)}/{len(summary)}`",
    ]

    if not best_ic.empty:
        top = best_ic.iloc[0]
        lines.extend(
            [
                f"- Best IC alpha: `{top['alpha']}` with `IC mean={fmt_num(top['ic_mean'])}`, `Sharpe(net)={fmt_num(top['sharpe_net'])}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Top Alphas By IC",
            top_table(
                best_ic,
                ["alpha", "ic_mean", "ic_ir", "annual_return_gross", "annual_return_net", "sharpe_net", "average_turnover"],
            ),
            "",
            "## Top Alphas By Net Return",
            top_table(
                best_net,
                ["alpha", "annual_return_net", "ic_mean", "sharpe_net", "max_drawdown_net", "average_turnover"],
            ),
            "",
            "## Validation Judgment",
            "This project now qualifies as a strong implementation and backtest framework for WorldQuant 101 on A-shares, but it does not yet qualify as a full paper-grade replication success if the accessible data window is short or restricted.",
            "A full replication claim requires long-horizon data, a large stable stock universe, explicit treatment of A-share trading constraints, and successful multi-factor validation under documented assumptions.",
            "",
            "## Current Limitations",
            "- The current JQData permission window may restrict the available history length. That directly weakens the statistical strength of any replication conclusion.",
            "- A-share shorting is limited in practice. The current long-short evaluation is best interpreted as a theoretical cross-sectional test, not a fully implementable retail trading strategy.",
            "- Some alphas are structurally high-turnover. Negative net results can come from transaction costs rather than formula bugs.",
            "- If the tested universe is still relatively small, cross-sectional signal stability may differ from the original paper context.",
        ]
    )
    if args.notes:
        lines.append(f"- {args.notes}")

    if errors is not None and not errors.empty:
        lines.extend(
            [
                "",
                "## Failed Alphas",
                errors.to_markdown(index=False),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    prices = load_csv(args.prices)
    meta = load_csv(args.meta)
    summary = load_csv(args.batch_summary)
    errors = load_csv(args.batch_errors) if args.batch_errors and Path(args.batch_errors).exists() else None
    report = build_report(args, prices, meta, summary, errors)
    Path(args.report_path).write_text(report, encoding="utf-8")
    print(f"Report written to {args.report_path}")


if __name__ == "__main__":
    main()
