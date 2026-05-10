# WorldQuant 101 A股因子复现与回测框架

这个项目实现了 `WorldQuant 101 Formulaic Alphas` 的 `101` 个公式因子，并在此基础上搭建了一套面向 A 股研究的真实数据导出、横截面回测与批量验证框架。

当前版本的目标不是“论文最终复现结论”，而是构建一套结构清晰、工程完整、适合展示在 GitHub 和简历中的量化研究项目。

## 项目亮点

- 实现 `101` 个 `WorldQuant 101` 公式因子
- 封装常用时序与截面算子，支持批量因子计算
- 支持 `JQData` 与 `Tushare Pro` 的 A 股数据导出
- 支持行业中性化、分层收益、多空组合、`IC/ICIR`、夏普、回撤、换手和交易成本分析
- 显式处理 A 股约束：`ST`、停牌、涨跌停、低价股、低成交额过滤
- 支持批量跑多个 alpha，并自动生成验证报告

## 当前定位

这套代码目前已经具备：

- 因子公式实现
- A 股真实数据接入
- 批量回测与验证
- 短窗口严格测试

但还不属于“长周期论文级严格复现”，因为最终学术级结论仍然依赖：

- 更长历史数据
- 更大且稳定的股票池
- 更强的稳健性检验

## 目录结构

```text
alphas/
  101alphas.md                 # 因子公式文本整理
  alpha101.py                  # 101 个 alpha 的实现
  base_ops.py                  # 公共算子
backtest_real_data.py          # 横截面回测主脚本
fetch_jqdata_a_share.py        # 从 JQData 导出 A 股数据
fetch_tushare_a_share.py       # 从 Tushare Pro 导出 A 股数据
generate_replication_report.py # 生成 Markdown 验证报告
main.py                        # 最小示例入口
docs/
  short_window_validation.md   # 短窗口验证摘要
requirements.txt
```

仓库默认不提交：

- 本地凭据
- 原始市场数据
- 回测输出结果
- 缓存和编辑器配置

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 环境变量

先复制模板：

```bash
cp .env.example .env
```

然后按需填写：

```text
JQDATA_USERNAME=你的聚宽账号
JQDATA_PASSWORD=你的聚宽密码
TUSHARE_TOKEN=你的Tushare Token
```

## 数据导出

### 1. 使用 JQData 导出 A 股数据

```bash
export JQDATA_USERNAME=你的聚宽账号
export JQDATA_PASSWORD=你的聚宽密码

python fetch_jqdata_a_share.py \
  --start 2025-02-03 \
  --end 2026-02-06 \
  --fq pre \
  --skip-st \
  --min-list-days 120 \
  --limit 120 \
  --batch-size 50 \
  --output-dir datas/jqdata_validation
```

输出文件：

- `prices.csv`
- `meta.csv`

当前 JQData 导出支持的字段包括：

- `open/high/low/close/volume/amount`
- `paused`
- `high_limit`
- `low_limit`
- `is_st`
- `cap`
- `circulating_cap`

### 2. 使用 Tushare Pro 导出 A 股数据

```bash
export TUSHARE_TOKEN=你的token

python fetch_tushare_a_share.py \
  --start 20230101 \
  --end 20251231 \
  --market 主板 \
  --min-list-days 120 \
  --adj qfq \
  --output-dir datas/tushare_export
```

## 回测示例

### 单个因子回测

```bash
python backtest_real_data.py \
  --source csv \
  --csv datas/jqdata_validation/prices.csv \
  --meta-csv datas/jqdata_validation/meta.csv \
  --alpha alpha_040 \
  --neutralize industry \
  --cost-bps 10 \
  --n-quantiles 5 \
  --min-amount 50000000 \
  --min-price 2 \
  --exclude-flat-bars \
  --output-dir outputs
```

### 批量因子回测

```bash
python backtest_real_data.py \
  --source csv \
  --csv datas/jqdata_validation/prices.csv \
  --meta-csv datas/jqdata_validation/meta.csv \
  --alphas alpha_001,alpha_002,alpha_003 \
  --neutralize industry \
  --cost-bps 10 \
  --n-quantiles 5 \
  --min-amount 50000000 \
  --min-price 2 \
  --exclude-flat-bars \
  --output-dir outputs_batch
```

## 回测指标

主要输出指标包括：

- `IC mean`
- `IC IR`
- `annual_return_gross`
- `annual_return_net`
- `sharpe_gross`
- `sharpe_net`
- `max_drawdown_gross`
- `max_drawdown_net`
- `average_turnover`

## 验证报告生成

```bash
python generate_replication_report.py \
  --prices datas/jqdata_validation/prices.csv \
  --meta datas/jqdata_validation/meta.csv \
  --batch-summary outputs_batch/batch_summary.csv \
  --report-path replication_report.md \
  --data-source JQData
```

## 当前已完成的验证

在最近一轮严格短窗口 A 股测试中，本项目已经完成：

- `101 / 101` 个因子的可执行验证
- 真实 A 股数据导出与接入
- 行业中性化横截面回测
- A 股交易约束过滤
- 批量因子结果汇总
- Markdown 验证报告输出

这说明当前仓库已经具备较完整的因子研究与工程展示价值。

## 局限性

- 当前最主要限制仍然是数据时间跨度不足
- A 股做长短组合更适合视为研究测试，不应直接等同于可实盘策略
- 部分因子天然高换手，对交易成本非常敏感
- 若要做论文级长期复现，还需要更长历史数据与更完整稳健性检验

## 下一版计划

当后续拿到更长历史数据后，可以继续补：

- 多年期长窗口复现
- 更大股票池验证
- benchmark 对比
- 分年度或分市场状态分析
- 更严格的稳健性与显著性检验
