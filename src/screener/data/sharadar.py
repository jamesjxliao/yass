"""Sharadar (Nasdaq Data Link) data provider.

Implements the DataProvider protocol on Sharadar's table-oriented API:
  SF1  — Core US Fundamentals (as-reported dimensions; ``datekey`` is the date
         the data became publicly available — used directly as PIT observed_at)
  SEP  — equity prices (split-adjusted close, ex-dividend, same convention as
         the FMP ``close`` this codebase has always used)
  SFP  — fund/ETF prices (benchmarks: SPY/QQQ/...)
  SP500 — index membership events since 1957 (survivorship-free)
  TICKERS — metadata (sectors, delisted flags)

Field parity with the FMP provider: every screening field except
``analyst_target``/``insider_buy_ratio`` (Sharadar has no analyst data; insider
data exists in SF2 but is not part of the screen) and ``beta`` (unused by all
signals/filters). All fields are computed from raw statement components, so
every value is auditable — see ``compute_arq_fields``.

Conventions:
  - ratios are single-quarter (matches the FMP quarterly key-metrics convention
    the signals were built on; z-scores are scale-invariant)
  - gross/op margins are annual-FY current vs prior (Piotroski criterion 8 is
    annual by design: annual cadence avoids noise-driven direction flips)
  - ``ev_to_sales`` approximates annualized sales as 4x the latest quarter
    (only consumed by non-production signals)
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import httpx
import polars as pl

from screener.data.cache import CacheManager
from screener.data.splits import heal_split_prices

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)  # api_key is a query param

BASE_URL = "https://data.nasdaq.com/api/v3/datatables"

# Benchmark/hedge ETFs live in SFP, not SEP
_FUND_TICKERS = {"SPY", "QQQ", "DIA", "IWM", "TLT", "HYG", "LQD", "GLD"}

# Secondary share classes whose primary class carries the fundamentals (SF1
# attaches statements to one class per company). Current-era: GOOG/FOX/NWS
# (production has always deduped these); historical-era members: CMCSK (CMCSA),
# DISCK (WBD), UA (UAA), TFCFA (TFCF). Keeping them would add unpickable
# null-fundamentals rows and double-count the company in the universe.
_SHARE_CLASS_DUPES = {"GOOG", "FOX", "NWS", "CMCSK", "DISCK", "UA", "TFCFA"}

_ARQ_COLS = (
    "ticker,dimension,calendardate,datekey,currentratio,debtusd,cashnequsd,"
    "ebitda,netinc,marketcap,fcf,capex,revenue,rnd,equity,assets,ebit,ebt,"
    "taxexp,invcap,sgna,sbcomp,ncfo,intangibles,receivables,inventory,"
    "payables,cor,ev,eps"
)
_ARY_COLS = "ticker,calendardate,datekey,grossmargin,opinc,revenue"


def _ratio(n, d):
    return (n / d) if (n is not None and d not in (None, 0)) else None


def compute_arq_fields(r: dict) -> dict[str, float | None]:
    """Screening fields from one SF1 ARQ row (production FMP conventions)."""
    out: dict[str, float | None] = {}
    out["roe"] = _ratio(r["netinc"], r["equity"])
    out["roa"] = _ratio(r["netinc"], r["assets"])
    tax = _ratio(r["taxexp"], r["ebt"])
    tax = min(max(tax, 0.0), 0.5) if tax is not None else 0.25
    nopat = r["ebit"] * (1 - tax) if r["ebit"] is not None else None
    out["roic"] = _ratio(nopat, r["invcap"])
    out["current_ratio"] = r["currentratio"]
    net_debt = (
        (r["debtusd"] - r["cashnequsd"])
        if None not in (r["debtusd"], r["cashnequsd"]) else None
    )
    out["net_debt_to_ebitda"] = _ratio(net_debt, r["ebitda"])
    out["earnings_yield"] = _ratio(r["netinc"], r["marketcap"])
    out["fcf_yield"] = _ratio(r["fcf"], r["marketcap"])
    capex = abs(r["capex"]) if r["capex"] is not None else None
    out["capex_to_revenue"] = _ratio(capex, r["revenue"])
    out["rd_to_revenue"] = _ratio(r["rnd"], r["revenue"])
    out["market_cap"] = r["marketcap"]
    out["sga_to_revenue"] = _ratio(r["sgna"], r["revenue"])
    out["sbc_to_revenue"] = _ratio(r["sbcomp"], r["revenue"])
    out["income_quality"] = _ratio(r["ncfo"], r["netinc"])
    out["intangibles_to_assets"] = _ratio(r["intangibles"], r["assets"])
    # Cash conversion cycle: DSO + DIO - DPO on the single quarter (~91.25 days)
    dso = _ratio(r["receivables"], r["revenue"])
    dio = _ratio(r["inventory"], r["cor"])
    dpo = _ratio(r["payables"], r["cor"])
    if None not in (dso, dio, dpo):
        out["cash_conversion_cycle"] = (dso + dio - dpo) * 91.25
    out["ev_to_sales"] = _ratio(r["ev"], 4 * r["revenue"] if r["revenue"] else None)
    return out


def _growth(cur, yr_ago):
    """YoY growth, FMP convention: only defined when the year-ago base is positive."""
    if cur is None or yr_ago is None or yr_ago <= 0:
        return None
    return cur / yr_ago - 1


class SharadarProvider:
    """Raw Nasdaq Data Link client (pagination, 429 backoff, batching)."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = httpx.Client(timeout=120)

    def _fetch_all(self, table: str, params: dict) -> tuple[list, list[str]]:
        rows: list = []
        columns: list[str] | None = None
        cursor = None
        while True:
            p = {"api_key": self._api_key, "qopts.per_page": 10000, **params}
            if cursor:
                p["qopts.cursor_id"] = cursor
            for attempt in range(5):
                resp = self._client.get(f"{BASE_URL}/{table}", params=p)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                break
            else:
                raise RuntimeError(f"{table}: persistent 429s")
            body = resp.json()
            d = body["datatable"]
            if columns is None:
                columns = [c["name"] for c in d["columns"]]
            rows.extend(d["data"])
            cursor = body.get("meta", {}).get("next_cursor_id")
            if not cursor:
                return rows, columns

    def _frame(self, table: str, params: dict) -> pl.DataFrame:
        rows, cols = self._fetch_all(table, params)
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows, schema=cols, orient="row", infer_schema_length=None)

    # --- universe / membership -------------------------------------------

    def get_universe(self, index: str) -> list[str]:
        if index != "sp500":
            raise ValueError(f"Sharadar provider supports sp500 only, got {index!r}")
        df = self._frame("SHARADAR/SP500", {"action": "current",
                                            "qopts.columns": "ticker"})
        tickers = sorted(set(df["ticker"].to_list()))
        # match production share-class dedup (keep GOOGL/FOXA/NWSA)
        return [t for t in tickers if t not in _SHARE_CLASS_DUPES]

    def get_membership_events(self) -> pl.DataFrame:
        """All add/remove events since 1957 plus the current roster."""
        return self._frame("SHARADAR/SP500",
                           {"qopts.columns": "date,action,ticker"})

    def get_delisted(self) -> list[str]:
        df = self._frame("SHARADAR/TICKERS",
                         {"table": "SEP", "isdelisted": "Y",
                          "qopts.columns": "ticker"})
        return df["ticker"].to_list() if not df.is_empty() else []

    def get_sectors(self, tickers: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for batch in _batched(tickers, 100):
            df = self._frame("SHARADAR/TICKERS",
                             {"table": "SF1", "ticker": ",".join(batch),
                              "qopts.columns": "ticker,sector"})
            if df.is_empty():
                continue
            for t, s in df.iter_rows():
                if s:
                    out[t] = s
        return out

    # --- prices ------------------------------------------------------------

    def get_prices(self, tickers: list[str], start: date, end: date) -> pl.DataFrame:
        """Daily bars from SEP, with SFP for fund/ETF tickers."""
        frames = []
        stocks = [t for t in tickers if t not in _FUND_TICKERS]
        funds = [t for t in tickers if t in _FUND_TICKERS]
        for table, group in (("SHARADAR/SEP", stocks), ("SHARADAR/SFP", funds)):
            for batch in _batched(group, 25):
                df = self._frame(table, {
                    "ticker": ",".join(batch),
                    "date.gte": str(start), "date.lte": str(end),
                    "qopts.columns": "ticker,date,open,high,low,close,volume",
                })
                if not df.is_empty():
                    frames.append(df.with_columns(pl.col("date").str.to_date()))
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames)

    def get_prices_single(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        return self.get_prices([ticker], start, end)

    # --- fundamentals -------------------------------------------------------

    def get_arq_history(self, tickers: list[str], since: date) -> pl.DataFrame:
        frames = []
        for batch in _batched(tickers, 50):
            df = self._frame("SHARADAR/SF1", {
                "ticker": ",".join(batch), "dimension": "ARQ",
                "calendardate.gte": str(since), "qopts.columns": _ARQ_COLS,
            })
            if not df.is_empty():
                frames.append(df)
        return pl.concat(frames) if frames else pl.DataFrame()

    def get_ary_history(self, tickers: list[str], since: date) -> pl.DataFrame:
        frames = []
        for batch in _batched(tickers, 50):
            df = self._frame("SHARADAR/SF1", {
                "ticker": ",".join(batch), "dimension": "ARY",
                "calendardate.gte": str(since), "qopts.columns": _ARY_COLS,
            })
            if not df.is_empty():
                frames.append(df)
        return pl.concat(frames) if frames else pl.DataFrame()


def _batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _snapshots_from_history(arq: pl.DataFrame, ary: pl.DataFrame) -> list[tuple]:
    """PIT tuples (ticker, field, value, report_date, source, observed_at) from
    SF1 history frames. Windowed fields (priors, YoY growth) are stamped with the
    CURRENT row's datekey — the date the comparison became computable."""
    snaps: list[tuple] = []
    if not arq.is_empty():
        arq = arq.sort(["ticker", "calendardate"])
        by_ticker: dict[str, list[dict]] = {}
        for r in arq.iter_rows(named=True):
            by_ticker.setdefault(r["ticker"], []).append(r)
        for t, rows in by_ticker.items():
            for i, r in enumerate(rows):
                fields = compute_arq_fields(r)
                prev = rows[i - 1] if i >= 1 else None
                yr = rows[i - 4] if i >= 4 else None
                if prev is not None:
                    fields["sga_to_revenue_prior"] = _ratio(prev["sgna"], prev["revenue"])
                if yr is not None:
                    fields["rev_growth_current"] = _growth(r["revenue"], yr["revenue"])
                    fields["eps_growth_current"] = _growth(r["eps"], yr["eps"])
                if i >= 5:
                    p, pyr = rows[i - 1], rows[i - 5]
                    fields["rev_growth_prior"] = _growth(p["revenue"], pyr["revenue"])
                    fields["eps_growth_prior"] = _growth(p["eps"], pyr["eps"])
                for field, value in fields.items():
                    if value is not None:
                        snaps.append((t, field, float(value), r["calendardate"],
                                      "sharadar_arq", r["datekey"]))
    if not ary.is_empty():
        ary = ary.sort(["ticker", "calendardate"])
        adf = ary.with_columns(
            _ratio_expr("opinc", "revenue").alias("op_margin"),
        ).with_columns([
            pl.col("grossmargin").shift(1).over("ticker").alias("gm_prior"),
            pl.col("op_margin").shift(1).over("ticker").alias("om_prior"),
        ])
        for r in adf.iter_rows(named=True):
            for field, value in (
                ("gross_margin_current", r["grossmargin"]),
                ("gross_margin_prior", r["gm_prior"]),
                ("op_margin_current", r["op_margin"]),
                ("op_margin_prior", r["om_prior"]),
            ):
                if value is not None:
                    snaps.append((r["ticker"], field, float(value),
                                  r["calendardate"], "sharadar_ary", r["datekey"]))
    return snaps


def _ratio_expr(n: str, d: str) -> pl.Expr:
    return (
        pl.when(pl.col(d).is_not_null() & (pl.col(d) != 0) & pl.col(n).is_not_null())
        .then(pl.col(n) / pl.col(d))
        .otherwise(None)
    )


class CachedSharadarProvider:
    """Sharadar provider with DuckDB-backed caching (mirrors CachedFMPProvider).

    - Fundamentals: per-ticker rows cached in api_cache (7-day TTL, versioned key)
    - Prices: incremental gap-fill into price_cache (immutable history)
    - Universe: 30-day TTL
    - PIT: bulk snapshot collection for fetch-history
    """

    _CACHE_VERSION = 1
    PROFILE_TTL_HOURS = 7 * 24
    UNIVERSE_TTL_HOURS = 30 * 24

    def __init__(self, sharadar: SharadarProvider, cache: CacheManager):
        self._api = sharadar
        self._cache = cache

    # --- universe ----------------------------------------------------------

    def get_universe(self, index: str) -> list[str]:
        data = self._cache.get_or_fetch(
            provider="sharadar",
            endpoint=f"universe/{index}",
            fetch_fn=lambda: [{"symbol": t} for t in self._api.get_universe(index)],
            ttl_hours=self.UNIVERSE_TTL_HOURS,
        )
        return [row["symbol"] for row in data]

    def get_delisted(self) -> list[str]:
        data = self._cache.get_or_fetch(
            provider="sharadar",
            endpoint="delisted",
            fetch_fn=lambda: [{"symbol": t} for t in self._api.get_delisted()],
            ttl_hours=self.UNIVERSE_TTL_HOURS,
        )
        return [row["symbol"] for row in data]

    def sync_membership(self) -> int:
        """Full survivorship-free sp500 membership from SP500 events.

        Unlike UniverseManager.sync (current-members diffing), this rebuilds the
        complete interval history from the event log — no PIT corruption risk."""
        from screener.data.universe import UniverseManager

        df = self._api.get_membership_events()
        events = df.filter(pl.col("action").is_in(["added", "removed"])).sort("date")
        current = set(df.filter(pl.col("action") == "current")["ticker"].to_list())
        stints: dict[str, list] = {}
        open_stint: dict[str, str] = {}
        for row in events.iter_rows(named=True):
            t, d, a = row["ticker"], row["date"], row["action"]
            if a == "added":
                open_stint.setdefault(t, d)
            else:
                start = open_stint.pop(t, "1957-03-04")
                stints.setdefault(t, []).append((start, d))
        for t, start in open_stint.items():
            stints.setdefault(t, []).append((start, None))
        for t in current:
            if t not in stints:
                stints.setdefault(t, []).append(("1957-03-04", None))
        records = [
            {"ticker": t, "added_date": s, "removed_date": e}
            for t, ss in stints.items() for s, e in ss
            if t not in _SHARE_CLASS_DUPES  # one class per company
        ]
        self._cache._conn.execute(
            "DELETE FROM universe_membership WHERE index_name = 'sp500'"
        )
        return UniverseManager(self._cache).bulk_load_history("sp500", records)

    # --- fundamentals (live screen) -----------------------------------------

    def _fundamentals_cache_key(self, ticker: str) -> str:
        return f"fundamentals/sharadar/v{self._CACHE_VERSION}/{ticker}"

    def get_fundamentals(self, tickers: list[str]) -> pl.DataFrame:
        cached_rows: list[dict] = []
        uncached: list[str] = []
        for t in tickers:
            hit = self._cache.get_or_fetch_if_cached(
                "sharadar", self._fundamentals_cache_key(t)
            )
            if hit:
                cached_rows.extend(hit)
            else:
                uncached.append(t)

        if uncached:
            logger.info("Fundamentals: %d cached, fetching %d from Sharadar",
                        len(tickers) - len(uncached), len(uncached))
            fresh = self._build_fundamentals_rows(uncached)
            for t in uncached:
                row = fresh.get(t)
                self._cache.get_or_fetch(
                    provider="sharadar",
                    endpoint=self._fundamentals_cache_key(t),
                    fetch_fn=lambda row=row: [row] if row else [],
                    ttl_hours=self.PROFILE_TTL_HOURS,
                )
                if row:
                    cached_rows.append(row)
        else:
            logger.info("Fundamentals: all %d cached", len(tickers))

        if not cached_rows:
            return pl.DataFrame()
        return pl.DataFrame(cached_rows, infer_schema_length=None)

    def _build_fundamentals_rows(self, tickers: list[str]) -> dict[str, dict]:
        """Latest screening row per ticker from SF1 (+ price tail from SEP)."""
        today = date.today()
        arq = self._api.get_arq_history(tickers, today - timedelta(days=30 * 30))
        ary = self._api.get_ary_history(tickers, today - timedelta(days=365 * 3))
        prices = self._api.get_prices(tickers, today - timedelta(days=45), today)

        rows: dict[str, dict] = {}
        if not arq.is_empty():
            arq = arq.sort(["ticker", "calendardate"])
            for t, grp in arq.group_by("ticker", maintain_order=True):
                hist = grp.to_dicts()
                r = hist[-1]
                row: dict = {"ticker": t[0] if isinstance(t, tuple) else t}
                row.update({k: v for k, v in compute_arq_fields(r).items()
                            if v is not None})
                if len(hist) >= 2:
                    p = hist[-2]
                    v = _ratio(p["sgna"], p["revenue"])
                    if v is not None:
                        row["sga_to_revenue_prior"] = v
                if len(hist) >= 5:
                    row["rev_growth_current"] = _growth(r["revenue"], hist[-5]["revenue"])
                    row["eps_growth_current"] = _growth(r["eps"], hist[-5]["eps"])
                if len(hist) >= 6:
                    row["rev_growth_prior"] = _growth(hist[-2]["revenue"], hist[-6]["revenue"])
                    row["eps_growth_prior"] = _growth(hist[-2]["eps"], hist[-6]["eps"])
                rows[row["ticker"]] = row
        if not ary.is_empty():
            ary = ary.sort(["ticker", "calendardate"])
            for t, grp in ary.group_by("ticker", maintain_order=True):
                hist = grp.to_dicts()
                ticker = t[0] if isinstance(t, tuple) else t
                if ticker not in rows:
                    continue
                cur = hist[-1]
                rows[ticker]["gross_margin_current"] = cur["grossmargin"]
                om = _ratio(cur["opinc"], cur["revenue"])
                if om is not None:
                    rows[ticker]["op_margin_current"] = om
                if len(hist) >= 2:
                    prev = hist[-2]
                    rows[ticker]["gross_margin_prior"] = prev["grossmargin"]
                    omp = _ratio(prev["opinc"], prev["revenue"])
                    if omp is not None:
                        rows[ticker]["op_margin_prior"] = omp
        if not prices.is_empty():
            tail = (prices.sort(["ticker", "date"])
                    .group_by("ticker", maintain_order=True)
                    .agg([pl.col("close").last().alias("close"),
                          pl.col("volume").tail(20).mean().alias("avg_volume_20d")]))
            for r in tail.iter_rows(named=True):
                if r["ticker"] in rows:
                    rows[r["ticker"]]["close"] = r["close"]
                    rows[r["ticker"]]["avg_volume_20d"] = r["avg_volume_20d"]
        return rows

    # --- prices (incremental gap-fill, mirrors CachedFMPProvider) -----------

    def get_prices(self, tickers: list[str], start: date, end: date) -> pl.DataFrame:
        to_fetch_full, gaps = [], []
        for ticker in tickers:
            cached_range = self._cache.get_cached_price_range(ticker)
            if cached_range is None:
                to_fetch_full.append(ticker)
                continue
            cached_start = date.fromisoformat(cached_range[0])
            cached_end = date.fromisoformat(cached_range[1])
            if start < cached_start:
                gaps.append((ticker, start, cached_start - timedelta(days=1)))
            if end > cached_end:
                gaps.append((ticker, cached_end + timedelta(days=1), end))

        if to_fetch_full:
            df = self._api.get_prices(to_fetch_full, start, end)
            if not df.is_empty():
                self._cache.store_prices(df, source="sharadar")
        # group gap fills by identical windows to batch them
        by_window: dict[tuple, list[str]] = {}
        for ticker, s, e in gaps:
            by_window.setdefault((s, e), []).append(ticker)
        for (s, e), group in by_window.items():
            df = self._api.get_prices(group, s, e)
            if not df.is_empty():
                self._cache.store_prices(df, source="sharadar")
        if to_fetch_full or gaps:
            logger.info("Prices: fetched %d full + %d gap windows",
                        len(to_fetch_full), len(by_window))
        result = self._cache.get_prices(tickers, str(start), str(end))
        # SEP closes are retroactively split-adjusted: a new split leaves every
        # cached row one adjustment behind the freshly fetched gap rows.
        return heal_split_prices(
            self._cache, result, tickers, start, end,
            fetch_single=self._api.get_prices_single, source="sharadar",
        )

    def get_prices_single(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        return self.get_prices([ticker], start, end)

    # --- PIT (fetch-history) -------------------------------------------------

    def collect_pit_snapshots_bulk(self, tickers: list[str], years: int = 10,
                                   progress=None) -> list[tuple]:
        """PIT snapshots for many tickers via batched table queries.

        Unlike FMP (7 calls per ticker), this needs ~2 calls per 50 tickers.
        ``progress`` is an optional callable(batch_index, n_batches)."""
        since = date.today() - timedelta(days=int(365.25 * years) + 400)
        snaps: list[tuple] = []
        batches = list(_batched(tickers, 50))
        for i, batch in enumerate(batches):
            arq = self._api.get_arq_history(batch, since)
            ary = self._api.get_ary_history(batch, since)
            snaps.extend(_snapshots_from_history(arq, ary))
            if progress:
                progress(i + 1, len(batches))
        return snaps

    def store_sectors(self, tickers: list[str]) -> int:
        sectors = self._api.get_sectors(tickers)
        for t, s in sectors.items():
            self._cache.store_sector(t, s)
        return len(sectors)
