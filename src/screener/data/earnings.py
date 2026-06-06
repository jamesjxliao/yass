from __future__ import annotations

import logging
import time
from datetime import date, datetime

import httpx

logger = logging.getLogger(__name__)

SEC_BASE = "https://data.sec.gov"
_RATE_INTERVAL = 0.12  # SEC allows 10 req/sec


class EarningsProvider:
    """Fetch actual quarterly EPS from SEC EDGAR XBRL and compute
    Standardized Unexpected Earnings (SUE) for post-earnings drift signal."""

    def __init__(self, user_agent: str):
        self._headers = {"User-Agent": user_agent}
        self._cik_map: dict[str, str] | None = None
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < _RATE_INTERVAL:
            time.sleep(_RATE_INTERVAL - elapsed)
        self._last_request = time.monotonic()

    def _load_cik_map(self) -> dict[str, str]:
        if self._cik_map is not None:
            return self._cik_map
        self._rate_limit()
        resp = httpx.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        self._cik_map = {
            v["ticker"]: str(v["cik_str"]).zfill(10)
            for v in resp.json().values()
        }
        return self._cik_map

    def _get_cik(self, ticker: str) -> str | None:
        return self._load_cik_map().get(ticker.upper())

    def get_quarterly_eps(self, ticker: str) -> list[dict]:
        """Fetch single-quarter diluted EPS from EDGAR XBRL.

        Returns list of {period_end, eps, filed} sorted by period_end desc.
        Deduplicates to the earliest filing per period.
        """
        cik = self._get_cik(ticker)
        if not cik:
            return []

        self._rate_limit()
        try:
            resp = httpx.get(
                f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json",
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, httpx.HTTPStatusError):
            logger.debug("EDGAR XBRL unavailable for %s (CIK %s)", ticker, cik)
            return []

        usgaap = data.get("facts", {}).get("us-gaap", {})
        eps_data = usgaap.get("EarningsPerShareDiluted", {})
        entries = eps_data.get("units", {}).get("USD/shares", [])

        if not entries:
            eps_data = usgaap.get("EarningsPerShareBasic", {})
            entries = eps_data.get("units", {}).get("USD/shares", [])

        quarterly: list[dict] = []
        for e in entries:
            if e.get("form") not in ("10-Q", "10-K"):
                continue
            start_str = e.get("start")
            if not start_str:
                continue
            try:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                end_dt = datetime.strptime(e["end"], "%Y-%m-%d")
            except ValueError:
                continue
            duration = (end_dt - start_dt).days
            if not (75 <= duration <= 105):
                continue
            quarterly.append({
                "period_end": e["end"],
                "eps": e["val"],
                "filed": e["filed"],
            })

        seen: dict[str, dict] = {}
        for e in quarterly:
            key = e["period_end"]
            if key not in seen or e["filed"] < seen[key]["filed"]:
                seen[key] = e

        return sorted(seen.values(), key=lambda x: x["period_end"], reverse=True)

    def compute_surprise(
        self, eps_history: list[dict],
    ) -> list[dict]:
        """Compute SUE (Standardized Unexpected Earnings) for each quarter.

        SUE = (EPS_q - EPS_q-4) / stdev(EPS_q - EPS_q-4)
        Uses seasonal random walk: expected EPS = same quarter last year.
        """
        by_period = {e["period_end"]: e for e in eps_history}
        periods = sorted(by_period.keys())

        raw_surprises: list[float] = []
        results: list[dict] = []

        for period in periods:
            entry = by_period[period]
            try:
                end_dt = datetime.strptime(period, "%Y-%m-%d")
                prior_end = date(end_dt.year - 1, end_dt.month, end_dt.day)
            except ValueError:
                continue

            best_match = None
            for p in periods:
                p_dt = datetime.strptime(p, "%Y-%m-%d").date()
                if abs((p_dt - prior_end).days) <= 10:
                    best_match = p
                    break

            if best_match is None:
                continue

            prior_eps = by_period[best_match]["eps"]
            if prior_eps == 0:
                continue

            raw_surprise = entry["eps"] - prior_eps
            raw_surprises.append(raw_surprise)

            results.append({
                "period_end": period,
                "filed": entry["filed"],
                "eps": entry["eps"],
                "eps_prior_year": prior_eps,
                "raw_surprise": raw_surprise,
            })

        if len(raw_surprises) < 4:
            return results

        import statistics
        std = statistics.stdev(raw_surprises)
        if std > 0:
            for r in results:
                r["sue"] = r["raw_surprise"] / std
        else:
            for r in results:
                r["sue"] = 0.0

        return results

    def get_surprise_snapshots(
        self, ticker: str,
    ) -> list[tuple[str, str, float, str, str, str]]:
        """Fetch EPS and compute SUE, returning PIT snapshot tuples.

        Returns list of (ticker, field, value, report_date, source, observed_at).
        """
        eps_history = self.get_quarterly_eps(ticker)
        if not eps_history:
            return []

        surprises = self.compute_surprise(eps_history)
        snapshots: list[tuple[str, str, float, str, str, str]] = []

        for s in surprises:
            if "sue" not in s:
                continue
            snapshots.append((
                ticker, "earnings_surprise",
                s["sue"], s["period_end"], "edgar", s["filed"],
            ))

        return snapshots
