import numpy as np
import pandas as pd

import alphas.alpha101 as alpha


def build_mock_data(num_days=252, seed=42):
    """生成一组满足 Alpha101 常见输入约束的模拟行情数据。"""
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start="2023-01-02", periods=num_days)
    ticks = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]

    base_price = pd.DataFrame(
        rng.uniform(80, 220, (num_days, len(ticks))),
        index=dates,
        columns=ticks,
    )
    daily_return = pd.DataFrame(
        rng.normal(0, 0.02, (num_days, len(ticks))),
        index=dates,
        columns=ticks,
    )

    close = base_price.mul((1 + daily_return).cumprod())
    open_ = close.shift(1).fillna(base_price).mul(
        1 + pd.DataFrame(rng.normal(0, 0.01, (num_days, len(ticks))), index=dates, columns=ticks)
    )

    intraday_spread = pd.DataFrame(
        rng.uniform(0.002, 0.03, (num_days, len(ticks))),
        index=dates,
        columns=ticks,
    )
    price_max = np.maximum(open_.values, close.values)
    price_min = np.minimum(open_.values, close.values)

    high = pd.DataFrame(
        price_max * (1 + intraday_spread.values),
        index=dates,
        columns=ticks,
    )
    low = pd.DataFrame(
        price_min * (1 - intraday_spread.values),
        index=dates,
        columns=ticks,
    )

    vwap_ratio = pd.DataFrame(
        rng.uniform(0.25, 0.75, (num_days, len(ticks))),
        index=dates,
        columns=ticks,
    )
    vwap = low + (high - low) * vwap_ratio

    volume = pd.DataFrame(
        rng.integers(1_000_000, 8_000_000, size=(num_days, len(ticks))),
        index=dates,
        columns=ticks,
    )
    cap = close * volume

    sector = pd.Series(
        {
            "AAPL": "Technology",
            "GOOG": "Technology",
            "MSFT": "Technology",
            "AMZN": "Consumer",
            "TSLA": "Automotive",
        }
    )
    industry = pd.Series(
        {
            "AAPL": "Consumer Electronics",
            "GOOG": "Internet Services",
            "MSFT": "Software",
            "AMZN": "E-Commerce",
            "TSLA": "EV",
        }
    )
    subindustry = pd.Series(
        {
            "AAPL": "Hardware",
            "GOOG": "Search",
            "MSFT": "Enterprise Software",
            "AMZN": "Online Retail",
            "TSLA": "Electric Vehicles",
        }
    )

    return {
        "open": open_,
        "close": close,
        "high": high,
        "low": low,
        "vwap": vwap,
        "volume": volume,
        "cap": cap,
        "sector": sector,
        "industry": industry,
        "subindustry": subindustry,
    }


def main():
    dummy_data = build_mock_data()
    alphas_inst = alpha.Alphas(dummy_data)
    print(alphas_inst.alpha_001().tail())


if __name__ == "__main__":
    main()
