import numpy as np
import pandas as pd
from .base_ops import *

EPS = 1e-12

class Alphas:
    def __init__(self, data):
        """
        data: 包含字段 DataFrame 的字典
        """
        self.open = data.get('open')
        self.close = data.get('close')
        self.high = data.get('high')
        self.low = data.get('low')
        self.volume = data.get('volume')
        self.vwap = data.get('vwap')
        self.returns = self.close.pct_change()
        self.cap = data.get('cap')
        self.sector = data.get('sector')
        self.industry = data.get('industry')
        self.subindustry = data.get('subindustry')

    def _like(self):
        return self.close

    def _const(self, value):
        return pd.DataFrame(value, index=self.close.index, columns=self.close.columns)

    def _adv(self, window):
        return ts_mean(self.volume, window)

    def _group(self, level):
        return getattr(self, level, None)

    def _indneutralize(self, df, level):
        group = self._group(level)
        if group is None:
            return df
        if isinstance(group, dict):
            group = pd.Series(group)
        if isinstance(group, pd.Series):
            group = group.reindex(df.columns)
            out = df.copy()
            for key in group.dropna().unique():
                cols = group[group == key].index.tolist()
                if cols:
                    out[cols] = out[cols].sub(out[cols].mean(axis=1), axis=0)
            return out
        if isinstance(group, pd.DataFrame) and group.shape == df.shape:
            out = df.copy()
            for idx in df.index:
                labels = group.loc[idx]
                for key in labels.dropna().unique():
                    cols = labels[labels == key].index.tolist()
                    if cols:
                        out.loc[idx, cols] = out.loc[idx, cols] - out.loc[idx, cols].mean()
            return out
        return df

    def _cap(self):
        if self.cap is not None:
            return self.cap
        return self.close * self.volume

    def _clean(self, df, fill=0.0):
        return sanitize(df).fillna(fill)

    def alpha_demo_rank(self):
        """示例 Alpha"""
        change_5d = delta(self.close, 5)
        return rank(change_5d)

    def alpha_001(self) -> pd.DataFrame:
        cond_val = self.close.where(self.returns >= 0, stddev(self.returns, 20))
        return rank(ts_argmax(signed_power(cond_val, 2.0), 5)) - 0.5

    def alpha_002(self):
        return self._clean(-1 * correlation(rank(delta(np.log(self.volume), 2)), rank((self.close - self.open) / self.open), 6))

    def alpha_003(self):
        return self._clean(-1 * correlation(rank(self.open), rank(self.volume), 10))

    def alpha_004(self):
        return -1 * ts_rank(rank(self.low), 9)

    def alpha_005(self):
        return rank(self.open - ts_mean(self.vwap, 10)) * (-1 * rank(self.close - self.vwap).abs())

    def alpha_006(self):
        return self._clean(-1 * correlation(self.open, self.volume, 10))

    def alpha_007(self):
        adv20 = self._adv(20)
        alpha = (-1 * ts_rank(delta(self.close, 7).abs(), 60)) * np.sign(delta(self.close, 7))
        alpha = alpha.where(adv20 < self.volume, -1.0)
        return alpha

    def alpha_008(self):
        inner = ts_sum(self.open, 5) * ts_sum(self.returns, 5)
        return -1 * rank(inner - delay(inner, 10))

    def alpha_009(self):
        d = delta(self.close, 1)
        return d.where((ts_min(d, 5) > 0) | (ts_max(d, 5) < 0), -1 * d)

    def alpha_010(self):
        d = delta(self.close, 1)
        return rank(d.where((ts_min(d, 4) > 0) | (ts_max(d, 4) < 0), -1 * d))

    def alpha_011(self):
        return (rank(ts_max(self.vwap - self.close, 3)) + rank(ts_min(self.vwap - self.close, 3))) * rank(delta(self.volume, 3))

    def alpha_012(self):
        return np.sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha_013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha_014(self):
        return (-1 * rank(delta(self.returns, 3))) * self._clean(correlation(self.open, self.volume, 10))

    def alpha_015(self):
        return -1 * ts_sum(rank(self._clean(correlation(rank(self.high), rank(self.volume), 3))), 3)

    def alpha_016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha_017(self):
        adv20 = self._adv(20)
        return (-1 * rank(ts_rank(self.close, 10))) * rank(delta(delta(self.close, 1), 1)) * rank(ts_rank(safe_div(self.volume, adv20), 5))

    def alpha_018(self):
        inner = stddev((self.close - self.open).abs(), 5) + (self.close - self.open) + self._clean(correlation(self.close, self.open, 10))
        return -1 * rank(inner)

    def alpha_019(self):
        return (-1 * np.sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) * (1 + rank(1 + ts_sum(self.returns, 250)))

    def alpha_020(self):
        return (-1 * rank(self.open - delay(self.high, 1))) * rank(self.open - delay(self.close, 1)) * rank(self.open - delay(self.low, 1))

    def alpha_021(self):
        avg8 = ts_mean(self.close, 8)
        std8 = stddev(self.close, 8)
        avg2 = ts_mean(self.close, 2)
        adv20 = self._adv(20)
        alpha = self._const(-1.0)
        alpha = alpha.where(~(avg2 < (avg8 - std8)), 1.0)
        alpha = alpha.where(~(safe_div(self.volume, adv20) >= 1), 1.0)
        alpha = alpha.where(~((avg8 + std8) < avg2), -1.0)
        return alpha

    def alpha_022(self):
        return -1 * delta(self._clean(correlation(self.high, self.volume, 5)), 5) * rank(stddev(self.close, 20))

    def alpha_023(self):
        return (-1 * delta(self.high, 2)).where(ts_mean(self.high, 20) < self.high, 0.0)

    def alpha_024(self):
        cond = safe_div(delta(ts_mean(self.close, 100), 100), delay(self.close, 100)) <= 0.05
        return (-1 * (self.close - ts_min(self.close, 100))).where(cond, -1 * delta(self.close, 3))

    def alpha_025(self):
        adv20 = self._adv(20)
        return rank(((-1 * self.returns) * adv20) * self.vwap * (self.high - self.close))

    def alpha_026(self):
        return -1 * ts_max(self._clean(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)), 3)

    def alpha_027(self):
        alpha = rank(ts_sum(self._clean(correlation(rank(self.volume), rank(self.vwap), 6)), 2) / 2.0)
        return alpha.where(alpha <= 0.5, -1.0).where(alpha > 0.5, 1.0)

    def alpha_028(self):
        adv20 = self._adv(20)
        return scale(self._clean(correlation(adv20, self.low, 5)) + ((self.high + self.low) / 2) - self.close)

    def alpha_029(self):
        part1 = rank(rank(scale(np.log(ts_sum(ts_min(rank(rank(-1 * rank(delta(self.close - 1, 5))), 2), 1), 1)))))
        return ts_min(product(part1, 1), 5) + ts_rank(delay(-1 * self.returns, 6), 5)

    def alpha_030(self):
        inner = np.sign(self.close - delay(self.close, 1)) + np.sign(delay(self.close, 1) - delay(self.close, 2)) + np.sign(delay(self.close, 2) - delay(self.close, 3))
        return (1.0 - rank(inner)) * ts_sum(self.volume, 5) / ts_sum(self.volume, 20)

    def alpha_031(self):
        adv20 = self._adv(20)
        part1 = rank(rank(rank(decay_linear(-1 * rank(rank(delta(self.close, 10))), 10))))
        part2 = rank(-1 * delta(self.close, 3))
        part3 = np.sign(scale(self._clean(correlation(adv20, self.low, 12))))
        return part1 + part2 + part3

    def alpha_032(self):
        return scale(ts_mean(self.close, 7) - self.close) + 20 * scale(self._clean(correlation(self.vwap, delay(self.close, 5), 230)))

    def alpha_033(self):
        return rank(-1 * (1 - (self.open / self.close)))

    def alpha_034(self):
        inner = safe_div(stddev(self.returns, 2), stddev(self.returns, 5))
        return rank((1 - rank(inner)) + (1 - rank(delta(self.close, 1))) + (1 - rank(ts_rank(safe_div(self.volume, self._adv(20)), 1))))

    def alpha_035(self):
        return ts_rank(self.volume, 32) * (1 - ts_rank((self.close + self.high) - self.low, 16)) * (1 - ts_rank(self.returns, 32))

    def alpha_036(self):
        adv20 = self._adv(20)
        part1 = 2.21 * rank(self._clean(correlation(self.close - self.open, delay(self.volume, 1), 15)))
        part2 = 0.7 * rank(self.open - self.close)
        part3 = 0.73 * rank(ts_rank(delay(-1 * self.returns, 6), 5))
        part4 = rank(self._clean(correlation(self.vwap, adv20, 6)).abs())
        part5 = 0.6 * rank((ts_mean(self.close, 200) - self.open) * (self.close - self.open))
        return part1 + part2 + part3 + part4 + part5

    def alpha_037(self):
        return rank(self._clean(correlation(delay(self.open - self.close, 1), self.close, 200))) + rank(self.open - self.close)

    def alpha_038(self):
        return (-1 * rank(ts_rank(self.close, 10))) * rank(self.close / self.open)

    def alpha_039(self):
        adv20 = self._adv(20)
        return (-1 * rank(delta(self.close, 7) * (1 - rank(decay_linear(safe_div(self.volume, adv20), 9))))) * (1 + rank(ts_sum(self.returns, 250)))

    def alpha_040(self):
        return (-1 * rank(stddev(self.high, 10))) * self._clean(correlation(self.high, self.volume, 10))

    def alpha_041(self):
        return np.sqrt(self.high * self.low) - self.vwap

    def alpha_042(self):
        return safe_div(rank(self.vwap - self.close), rank(self.vwap + self.close))

    def alpha_043(self):
        return ts_rank(safe_div(self.volume, self._adv(20)), 20) * ts_rank(-1 * delta(self.close, 7), 8)

    def alpha_044(self):
        return -1 * self._clean(correlation(self.high, rank(self.volume), 5))

    def alpha_045(self):
        part1 = rank(ts_mean(delay(self.close, 5), 20))
        part2 = self._clean(correlation(self.close, self.volume, 2))
        part3 = rank(self._clean(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))
        return -1 * part1 * part2 * part3

    def alpha_046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = -1 * (self.close - delay(self.close, 1))
        alpha = alpha.where(~(inner < 0), 1.0)
        alpha = alpha.where(~(inner > 0.25), -1.0)
        return alpha

    def alpha_047(self):
        adv20 = self._adv(20)
        return (((rank(1 / self.close) * self.volume) / adv20) * ((self.high * rank(self.high - self.close)) / ts_mean(self.high, 5))) - rank(self.vwap - delay(self.vwap, 5))

    def alpha_048(self):
        num = self._clean(correlation(delta(self.close, 1), delta(delay(self.close, 1), 1), 250)) * delta(self.close, 1)
        num = safe_div(num, self.close)
        num = self._indneutralize(num, 'subindustry')
        den = ts_sum(safe_div(delta(self.close, 1), delay(self.close, 1)) ** 2, 250)
        return safe_div(num, den)

    def alpha_049(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = -1 * (self.close - delay(self.close, 1))
        return alpha.where(~(inner < -0.1), 1.0)

    def alpha_050(self):
        return -1 * ts_max(rank(self._clean(correlation(rank(self.volume), rank(self.vwap), 5))), 5)

    def alpha_051(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = -1 * (self.close - delay(self.close, 1))
        return alpha.where(~(inner < -0.05), 1.0)

    def alpha_052(self):
        return ((-1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5)) * rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)) * ts_rank(self.volume, 5)

    def alpha_053(self):
        return -1 * delta(safe_div((self.close - self.low) - (self.high - self.close), self.close - self.low), 9)

    def alpha_054(self):
        return safe_div(-1 * (self.low - self.close) * (self.open ** 5), (self.low - self.high) * (self.close ** 5))

    def alpha_055(self):
        inner = safe_div(self.close - ts_min(self.low, 12), ts_max(self.high, 12) - ts_min(self.low, 12))
        return -1 * self._clean(correlation(rank(inner), rank(self.volume), 6))

    def alpha_056(self):
        return 0 - (rank(safe_div(ts_sum(self.returns, 10), ts_sum(ts_sum(self.returns, 2), 3))) * rank(self.returns * self._cap()))

    def alpha_057(self):
        return 0 - safe_div(self.close - self.vwap, decay_linear(rank(ts_argmax(self.close, 30)), 2))

    def alpha_058(self):
        neutral_vwap = self._indneutralize(self.vwap, 'sector')
        corr = self._clean(correlation(neutral_vwap, self.volume, 3.92795))
        return -1 * ts_rank(decay_linear(corr, 7.89291), 5.50322)

    def alpha_059(self):
        neutral_vwap = self._indneutralize((self.vwap * 0.728317) + (self.vwap * (1 - 0.728317)), 'industry')
        corr = self._clean(correlation(neutral_vwap, self.volume, 4.25197))
        return -1 * ts_rank(decay_linear(corr, 16.2289), 8.19648)

    def alpha_060(self):
        inner = safe_div(((self.close - self.low) - (self.high - self.close)) * self.volume, self.high - self.low)
        return 0 - (2 * scale(rank(inner)) - scale(rank(ts_argmax(self.close, 10))))

    def alpha_061(self):
        return bool_to_float(rank(self.vwap - ts_min(self.vwap, 16.1219)) < rank(self._clean(correlation(self.vwap, self._adv(180), 17.9282))))

    def alpha_062(self):
        adv20 = self._adv(20)
        left = rank(self._clean(correlation(self.vwap, ts_sum(adv20, 22.4101), 9.91009)))
        right = rank((rank(self.open) + rank(self.open)) < (rank((self.high + self.low) / 2) + rank(self.high)))
        return -1 * bool_to_float(left < right)

    def alpha_063(self):
        left = rank(decay_linear(delta(self._indneutralize(self.close, 'industry'), 2.25164), 8.22237))
        mix = (self.vwap * 0.318108) + (self.open * (1 - 0.318108))
        right = rank(decay_linear(self._clean(correlation(mix, ts_sum(self._adv(180), 37.2467), 13.557)), 12.2883))
        return -1 * (left - right)

    def alpha_064(self):
        left = rank(self._clean(correlation(ts_sum((self.open * 0.178404) + (self.low * (1 - 0.178404)), 13.6971), ts_sum(self._adv(120), 13.6971), 8.62571)))
        right = rank(delta((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404)), 1.46063))
        return -1 * bool_to_float(left < right)

    def alpha_065(self):
        left = rank(self._clean(correlation((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205)), ts_sum(self._adv(60), 8.6911), 6.40374)))
        right = rank(self.open - ts_min(self.open, 13.635))
        return -1 * bool_to_float(left < right)

    def alpha_066(self):
        left = rank(decay_linear(delta(self.vwap, 3.51013), 7.23052))
        inner = safe_div(((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap, self.open - ((self.high + self.low) / 2))
        right = ts_rank(decay_linear(inner, 11.4157), 6.72611)
        return -1 * (left + right)

    def alpha_067(self):
        left = rank(self.high - ts_min(self.high, 2.14593))
        right = rank(self._clean(correlation(self._indneutralize(self.vwap, 'sector'), self._indneutralize(self._adv(20), 'subindustry'), 6.02936)))
        return -1 * (left ** right)

    def alpha_068(self):
        left = ts_rank(self._clean(correlation(rank(self.high), rank(self._adv(15)), 8.91644)), 13.9333)
        right = rank(delta((self.close * 0.518371) + (self.low * (1 - 0.518371)), 1.06157))
        return -1 * bool_to_float(left < right)

    def alpha_069(self):
        left = rank(ts_max(delta(self._indneutralize(self.vwap, 'industry'), 2.72412), 4.79344))
        right = ts_rank(self._clean(correlation((self.close * 0.490655) + (self.vwap * (1 - 0.490655)), self._adv(20), 4.92416)), 9.0615)
        return -1 * (left ** right)

    def alpha_070(self):
        left = rank(delta(self.vwap, 1.29456))
        right = ts_rank(self._clean(correlation(self._indneutralize(self.close, 'industry'), self._adv(50), 17.8256)), 17.9171)
        return -1 * (left ** right)

    def alpha_071(self):
        part1 = ts_rank(decay_linear(self._clean(correlation(ts_rank(self.close, 3.43976), ts_rank(self._adv(180), 12.0647), 18.0175)), 4.20501), 15.6948)
        part2 = ts_rank(decay_linear(rank((self.low + self.open) - (self.vwap + self.vwap)) ** 2, 16.4662), 4.4388)
        return max_df(part1, part2)

    def alpha_072(self):
        left = rank(decay_linear(self._clean(correlation((self.high + self.low) / 2, self._adv(40), 8.93345)), 10.1519))
        right = rank(decay_linear(self._clean(correlation(ts_rank(self.vwap, 3.72469), ts_rank(self.volume, 18.5188), 6.86671)), 2.95011))
        return safe_div(left, right)

    def alpha_073(self):
        part1 = rank(decay_linear(delta(self.vwap, 4.72775), 2.91864))
        base = (self.open * 0.147155) + (self.low * (1 - 0.147155))
        part2 = ts_rank(decay_linear(-1 * safe_div(delta(base, 2.03608), base), 3.33829), 16.7411)
        return -1 * max_df(part1, part2)

    def alpha_074(self):
        left = rank(self._clean(correlation(self.close, ts_sum(self._adv(30), 37.4843), 15.1365)))
        right = rank(self._clean(correlation(rank((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661))), rank(self.volume), 11.4791)))
        return -1 * bool_to_float(left < right)

    def alpha_075(self):
        left = rank(self._clean(correlation(self.vwap, self.volume, 4.24304)))
        right = rank(self._clean(correlation(rank(self.low), rank(self._adv(50)), 12.4413)))
        return -1 * bool_to_float(left < right)

    def alpha_076(self):
        part1 = rank(decay_linear(delta(self.vwap, 1.24383), 11.8259))
        corr = self._clean(correlation(self._indneutralize(self.low, 'sector'), self._adv(81), 8.14941))
        part2 = ts_rank(decay_linear(ts_rank(corr, 19.569), 17.1543), 19.383)
        return -1 * max_df(part1, part2)

    def alpha_077(self):
        part1 = rank(decay_linear((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high), 20.0451))
        part2 = rank(decay_linear(self._clean(correlation((self.high + self.low) / 2, self._adv(40), 3.1614)), 5.64125))
        return -1 * min_df(part1, part2)

    def alpha_078(self):
        left = rank(self._clean(correlation(ts_sum((self.low * 0.352233) + (self.vwap * (1 - 0.352233)), 19.7428), ts_sum(self._adv(40), 19.7428), 6.83313)))
        right = rank(self._clean(correlation(rank(self.vwap), rank(self.volume), 5.77492)))
        return left ** right

    def alpha_079(self):
        left = rank(delta(self._indneutralize((self.close * 0.60733) + (self.open * (1 - 0.60733)), 'sector'), 1.23438))
        right = rank(self._clean(correlation(ts_rank(self.vwap, 3.60973), ts_rank(self._adv(150), 9.18637), 14.6644)))
        return -1 * bool_to_float(left < right)

    def alpha_080(self):
        left = rank(np.sign(delta(self._indneutralize((self.open * 0.868128) + (self.high * (1 - 0.868128)), 'industry'), 4.04545)))
        right = ts_rank(self._clean(correlation(self.high, self._adv(10), 5.11456)), 5.53756)
        return -1 * (left ** right)

    def alpha_081(self):
        left = rank(np.log(product(rank(rank(self._clean(correlation(self.vwap, ts_sum(self._adv(10), 49.6054), 8.47743))) ** 4), 14.9655)))
        right = rank(self._clean(correlation(rank(self.vwap), rank(self.volume), 5.07914)))
        return -1 * bool_to_float(left < right)

    def alpha_082(self):
        part1 = rank(decay_linear(delta(self.open, 1.46063), 14.8717))
        corr = self._clean(correlation(self._indneutralize(self.volume, 'sector'), (self.open * 0.634196) + (self.open * (1 - 0.634196)), 17.4842))
        part2 = ts_rank(decay_linear(corr, 6.92131), 13.4283)
        return -1 * min_df(part1, part2)

    def alpha_083(self):
        numerator = rank(delay(safe_div(self.high - self.low, ts_mean(self.close, 5)), 2)) * rank(rank(self.volume))
        denominator = safe_div(safe_div(self.high - self.low, ts_mean(self.close, 5)), self.vwap - self.close)
        return safe_div(numerator, denominator)

    def alpha_084(self):
        return signed_power(ts_rank(self.vwap - ts_max(self.vwap, 15.3217), 20.7127), delta(self.close, 4.96796))

    def alpha_085(self):
        left = rank(self._clean(correlation((self.high * 0.876703) + (self.close * (1 - 0.876703)), self._adv(30), 9.61331)))
        right = rank(self._clean(correlation(ts_rank((self.high + self.low) / 2, 3.70596), ts_rank(self.volume, 10.1595), 7.11408)))
        return left ** right

    def alpha_086(self):
        left = ts_rank(self._clean(correlation(self.close, ts_sum(self._adv(20), 14.7444), 6.00049)), 20.4195)
        right = rank((self.open + self.close) - (self.vwap + self.open))
        return -1 * bool_to_float(left < right)

    def alpha_087(self):
        part1 = rank(decay_linear(delta((self.close * 0.369701) + (self.vwap * (1 - 0.369701)), 1.91233), 2.65461))
        corr = self._clean(correlation(self._indneutralize(self._adv(81), 'industry'), self.close, 13.4132)).abs()
        part2 = ts_rank(decay_linear(corr, 4.89768), 14.4535)
        return -1 * max_df(part1, part2)

    def alpha_088(self):
        part1 = rank(decay_linear((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close)), 8.06882))
        part2 = ts_rank(decay_linear(self._clean(correlation(ts_rank(self.close, 8.44728), ts_rank(self._adv(60), 20.6966), 8.01266)), 6.65053), 2.61957)
        return -1 * min_df(part1, part2)

    def alpha_089(self):
        part1 = ts_rank(decay_linear(self._clean(correlation((self.low * 0.967285) + (self.low * (1 - 0.967285)), self._adv(10), 6.94279)), 5.51607), 3.79744)
        part2 = ts_rank(decay_linear(delta(self._indneutralize(self.vwap, 'industry'), 3.48158), 10.1466), 15.3012)
        return part1 - part2

    def alpha_090(self):
        left = rank(self.close - ts_max(self.close, 4.66719))
        right = ts_rank(self._clean(correlation(self._indneutralize(self._adv(40), 'subindustry'), self.low, 5.38375)), 3.21856)
        return -1 * (left ** right)

    def alpha_091(self):
        corr1 = self._clean(correlation(self._indneutralize(self.close, 'industry'), self.volume, 9.74928))
        left = ts_rank(decay_linear(decay_linear(corr1, 16.398), 3.83219), 4.8667)
        right = rank(decay_linear(self._clean(correlation(self.vwap, self._adv(30), 4.01303)), 2.6809))
        return -1 * (left - right)

    def alpha_092(self):
        part1 = ts_rank(decay_linear(bool_to_float((((self.high + self.low) / 2) + self.close) < (self.low + self.open)), 14.7221), 18.8683)
        part2 = ts_rank(decay_linear(self._clean(correlation(rank(self.low), rank(self._adv(30)), 7.58555)), 6.94024), 6.80584)
        return min_df(part1, part2)

    def alpha_093(self):
        left = ts_rank(decay_linear(self._clean(correlation(self._indneutralize(self.vwap, 'industry'), self._adv(81), 17.4193)), 19.848), 7.54455)
        right = rank(decay_linear(delta((self.close * 0.524434) + (self.vwap * (1 - 0.524434)), 2.77377), 16.2664))
        return safe_div(left, right)

    def alpha_094(self):
        left = rank(self.vwap - ts_min(self.vwap, 11.5783))
        right = ts_rank(self._clean(correlation(ts_rank(self.vwap, 19.6462), ts_rank(self._adv(60), 4.02992), 18.0926)), 2.70756)
        return -1 * (left ** right)

    def alpha_095(self):
        left = rank(self.open - ts_min(self.open, 12.4105))
        right = ts_rank(rank(self._clean(correlation(ts_sum((self.high + self.low) / 2, 19.1351), ts_sum(self._adv(40), 19.1351), 12.8742))) ** 5, 11.7584)
        return bool_to_float(left < right)

    def alpha_096(self):
        part1 = ts_rank(decay_linear(self._clean(correlation(rank(self.vwap), rank(self.volume), 3.83878)), 4.16783), 8.38151)
        corr = self._clean(correlation(ts_rank(self.close, 7.45404), ts_rank(self._adv(60), 4.13242), 3.65459))
        part2 = ts_rank(decay_linear(ts_argmax(corr, 12.6556), 14.0365), 13.4143)
        return -1 * max_df(part1, part2)

    def alpha_097(self):
        left = rank(decay_linear(delta(self._indneutralize((self.low * 0.721001) + (self.vwap * (1 - 0.721001)), 'industry'), 3.3705), 20.4523))
        corr = self._clean(correlation(ts_rank(self.low, 7.87871), ts_rank(self._adv(60), 17.255), 4.97547))
        right = ts_rank(decay_linear(ts_rank(corr, 18.5925), 15.7152), 6.71659)
        return -1 * (left - right)

    def alpha_098(self):
        left = rank(decay_linear(self._clean(correlation(self.vwap, ts_sum(self._adv(5), 26.4719), 4.58418)), 7.18088))
        corr = self._clean(correlation(rank(self.open), rank(self._adv(15)), 20.8187))
        right = rank(decay_linear(ts_rank(ts_argmin(corr, 8.62571), 6.95668), 8.07206))
        return left - right

    def alpha_099(self):
        left = rank(self._clean(correlation(ts_sum((self.high + self.low) / 2, 20.8187), ts_sum(self._adv(60), 20.8187), 8.62571)))
        right = rank(self._clean(correlation(self.low, self.volume, 6.28259)))
        return -1 * bool_to_float(left < right)

    def alpha_100(self):
        inner1 = rank(safe_div(((self.close - self.low) - (self.high - self.close)) * self.volume, self.high - self.low))
        inner1 = self._indneutralize(self._indneutralize(inner1, 'subindustry'), 'subindustry')
        inner2 = self._clean(correlation(self.close, rank(self._adv(20)), 5)) - rank(ts_argmin(self.close, 30))
        inner2 = self._indneutralize(inner2, 'subindustry')
        return 0 - (((1.5 * scale(inner1)) - scale(inner2)) * safe_div(self.volume, self._adv(20)))

    def alpha_101(self):
        return safe_div(self.close - self.open, (self.high - self.low) + 0.001)


if __name__ == "__main__":
    print("Alpha101 模块已加载")
