# 加密永续合约：在 qlib 里研究、回测、出实盘目标

一条链路，全部在 qlib 里：

| 环节 | 用什么 | 产出 |
| --- | --- | --- |
| 数据 | `jarvis-data sync` 后 `jarvis-data dump qlib` | `~/.qlib/qlib_data/crypto_um`：日线、资金费、OI、按时点的 `top100.txt` |
| 特征、模型 | qlib 的 DataHandler（如 Alpha158）和模型（如 LightGBM） | `pred.pkl`：每天每个币的分数 |
| 回测 | `qlib.contrib.crypto.PerpPortAnaRecord`（qrun 里）或 `backtest_long_short`（notebook 里） | 多空组合的日报表、持仓权重、逐笔成交 |
| 实盘 | `python -m qlib.contrib.crypto.live` | 每天一份目标权重 `targets-<日期>.json` |

回测和实盘用同一个 `LongShortPortfolio` 从分数算目标权重，实盘的分数也由同一个训练结果、同一套数据集配置算出：
实盘某一天的目标，就是回测在那一天会持有的组合。

## 研究：跑一次 qrun

```bash
qrun examples/crypto/workflow_config_lightgbm_Alpha158_crypto.yaml
```

配置里要按自己的情况改的：

| 设置 | 说明 |
| --- | --- |
| `qlib_init.region: crypto` | 不按手数取整、没有涨跌停、`risk_analysis` 按一年 365 天年化 |
| `market: top100` | `dump qlib` 写的按时点 universe；训练、预测、选币都只用当天在 universe 里的币 |
| `label` | `Ref($close,-2)/Ref($close,-1)-1-Ref($funding,-2)`：t 日收盘的分数在 t+1 收盘成交，持有到 t+2 收盘，多头付当天的资金费、空头收 |
| `portfolio` | `quantile`（或 `n_long`/`n_short`）、`gross`/`net`（多空各占多少权益）、`weighting`、`buffer`（降换手）、`max_weight`、`min_names` |
| `costs` | `fee`（可用 `fees` 按币覆盖）、`spread_tiers`（按 30 日成交额分档的滑点）、`impact`（平方根冲击）、`max_participation`（单日最多占成交额的比例） |
| `backtest.account` | 回测本金（USDT）。冲击成本随它变大，设成实际要交易的资金 |
| `backtest.delay`、`deal_price` | 默认 `1` 和 `close`，和 label 一致；`vwap` 表示在 t+1 日内按 vwap 成交 |

结果在 recorder 的 `portfolio_analysis/` 下：

```python
from qlib.workflow import R
rec = R.get_recorder(recorder_id="...", experiment_name="workflow")
report = rec.load_object("portfolio_analysis/report_normal_1day.pkl")    # 每日：account return cost funding turnover 敞口
weights = rec.load_object("portfolio_analysis/weights_1day.pkl")         # 每日收盘后的持仓权重
trades = rec.load_object("portfolio_analysis/trades_1day.pkl")           # 每笔成交和成本
analysis = rec.load_object("portfolio_analysis/port_analysis_1day.pkl")  # 年化、IR、回撤（365 天）

from qlib.contrib.report import analysis_position
analysis_position.report_graph(report)                                   # qlib 自带的收益图
```

在 notebook 里也可以直接回测任何分数，不必经过 qrun：

```python
from qlib.contrib.crypto import CostModel, LongShortPortfolio, backtest_long_short
result = backtest_long_short(
    pred,                                                  # (datetime, instrument) 的分数
    portfolio=LongShortPortfolio(quantile=0.1, buffer=2),
    costs=CostModel(fee=0.0005, impact=0.5, max_participation=0.02),
    account=1_000_000,
)
result.summary()       # 年化、IR、回撤、成本、资金费、换手、敞口
```

## 回测口径

分数并列：回测报表有 `tied_boundary` 列，标出多头或空头边缘有并列分数的调仓日，并在日志里给出天数。并列的币按币名字母序入选，
所以这些天的持仓不是模型选的。模型要有足够多的不同输出值（树够多、特征够多、训练样本够多）；这一列大量为 True 时，
先改模型，不要看回测收益。

每天按这个顺序：

1. **估值**：持仓按收盘价重估。某天没有收盘价（没有 K 线或没有成交），沿用上一个价格，这一天也不能交易。
2. **资金费**：持有过这一天的仓位付 `名义价值 × $funding`（为正时多头付、空头收）。`$funding` 为 NaN（当月还没发布、缺记录）时不扣，
   计入报表的 `funding_missing`；设 `funding_missing: raise` 则直接报错。
3. **调仓**：按决策日的分数算目标权重，目标名义价值 = 权重 × 当时权益，减去现有仓位就是要成交的量。当天不能交易的币保持原仓位；
   当天下架的币不进入目标。`max_participation` 限制单日成交量，超出的部分留到之后几天。
4. **下架**：上市区间最后一天还持有的仓位，按当天收盘价平掉（交易所结算）。
5. **成本**：每笔成交付 `fee + spread(30 日成交额) + impact × σ × sqrt(成交额 / 30 日成交额)`，σ 是前 30 天日收益的标准差。

报表的 `return` 是价格和资金费的收益（扣费前，qlib 的约定），`cost` 是交易成本，`return - cost` 是账户收益；
`account` 是权益。

没有模拟的：单个仓位的保证金和强平（按全仓、1–2 倍杠杆理解）、交易所的最小下单量和数量精度、资金费按当天开始时的名义价值计
（不随日内价格变化）、下架结算价用最后一个收盘价近似、做空没有借币限制。默认成本是起点，不是测量值，要用自己的成交校准。

## 实盘：每天出目标权重

```bash
python -m qlib.contrib.crypto.live \
    --provider-uri ~/.qlib/qlib_data/crypto_um \
    --mlruns ./mlruns --experiment workflow --recorder <训练的 recorder id> \
    --out ./targets --current ./current_weights.json --equity 100000 --trusted
```

它做的事：

1. 读 `--recorder` 那次训练：模型、数据集配置、回测用的组合设置。不给 `--recorder` 时用 experiment 里最新一次完成的训练；
   实盘要固定 `--recorder`，否则在同一个 experiment 里新跑的研究会被直接拿去交易。
   模型和数据集是 pickle，qlib 只在 `--trusted` 时加载，表示这是你自己训练的。
2. 用同一套数据集配置给决策日打分（默认是数据的最后一天），取前 `--lookback-days`（400）天算特征。
3. 检查，任何一条不满足就**不写文件**、退出码 2：
   - 决策日不能比今天（UTC）早超过 `--max-age-days`（2）天：说明 sync 没跑；
   - universe 成员里至少 `--min-coverage`（90%）当天有收盘价、有分数：说明 dump 不完整；
   - 任何一个原始特征在成员中 NaN 的比例不超过 `--max-feature-nan`（50%）：说明某个字段还没有数据（例如用 `$funding` 做特征，
     而当月资金费还没发布）；
   - 多头或空头那一侧的边缘不能有并列的分数（模型输出的不同取值太少，如只有几棵树）：并列的币由币名字母序决定谁入选，
     不是模型决定的。`--allow-ties` 可强制放行。
4. 用回测的 `LongShortPortfolio` 算目标（`--current` 是现在实际持有的权重，供 `buffer` 用），原子写入
   `targets-<决策日>.json` 和 `latest.json`。

输出：

```json
{
  "decision_date": "2026-11-04",
  "trade": {"deal_price": "close", "at": "2026-11-06T00:00:00Z"},
  "generated_at": "2026-11-05T09:12:03Z",
  "calendar_end": "2026-11-04",
  "recorder": {"id": "…", "experiment_id": "…", "name": "…"},
  "portfolio": {"quantile": 0.1, "gross": 1.0, "net": 0.0, "buffer": 2, "…": "…"},
  "weights": {"BTCUSDT": 0.05, "XYZUSDT": -0.05, "…": 0},
  "gross": 1.0, "net": 0.0, "n_long": 10, "n_short": 10,
  "equity": 100000, "notional": {"BTCUSDT": 5000.0, "…": 0},
  "checks": {"universe_members": 100, "priced": 100, "scored": 100, "worst_feature": {"name": "…", "nan_share": 0.0}}
}
```

权重是占权益的比例，正为多、负为空，不在 `weights` 里的币目标为 0。`trade.at` 是回测假设的成交时刻：决策日 d 的目标在
d+1 的收盘、也就是 d+2 日 00:00 UTC 成交。在别的时刻成交就是另一个策略，实盘和回测会有偏差。

每天的顺序（UTC）：

1. Vision 发布前一天的数据后：`jarvis-data sync`，然后 `jarvis-data dump qlib --out ~/.qlib/qlib_data/crypto_um --top 100`。
2. `python -m qlib.contrib.crypto.live …`。
3. 执行端读 `latest.json`：
   - 核对 `decision_date` 是不是预期的那一天、`generated_at` 是不是今天；不是就不交易。
   - 退出码 2 或文件没更新：保持现有仓位，不要按旧目标重复下单。
   - 把 `notional`（或 `weights × 实际权益`）换算成合约数量，按交易所的数量精度和最小名义价值取整，在 `trade.at` 附近成交。
   - 把成交后的实际权重写回 `--current` 的文件，第二天用。

重新训练：定期（例如每月）再跑一次 qrun，看过回测结果后，再把实盘命令的 `--recorder` 换成新的那一次。

## 上线前

- 成本：用几天的真实成交校准 `fee`、`spread_tiers`、`impact`，`account` 设成实际资金。
- 特征：只用实盘当天能拿到的字段。Vision 的资金费按月发布，`$funding` 当月是 NaN；metrics 也晚一天。label 用 `$funding` 没问题，
  特征用它会被第 3 步的检查拦下。
- 先只出目标、不下单，跑几天，看每天的目标、换手和回测是否一致。
- 时间一律用 UTC。
