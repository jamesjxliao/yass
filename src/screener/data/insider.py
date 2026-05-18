from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from xml.etree import ElementTree

import httpx
import polars as pl

logger = logging.getLogger(__name__)

SEC_BASE = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_RATE_INTERVAL = 0.12  # SEC allows 10 req/sec


class InsiderProvider:
    """Fetch real open-market insider purchases from SEC EDGAR Form 4 filings."""

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
        )
        resp.raise_for_status()
        self._cik_map = {
            v["ticker"]: str(v["cik_str"]) for v in resp.json().values()
        }
        return self._cik_map

    def _get_cik(self, ticker: str) -> str | None:
        return self._load_cik_map().get(ticker.upper())

    def _get_form4_accessions(
        self, cik: str, after_date: date
    ) -> list[dict[str, str]]:
        padded = cik.zfill(10)
        self._rate_limit()
        resp = httpx.get(
            f"{SEC_BASE}/submissions/CIK{padded}.json",
            headers=self._headers,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form != "4":
                continue
            if dates[i] < str(after_date):
                break
            doc = docs[i]
            if doc.startswith("xslF345X") and "/" in doc:
                doc = doc.split("/", 1)[1]
            results.append({
                "accession": accessions[i].replace("-", ""),
                "document": doc,
                "filing_date": dates[i],
            })
        return results

    def _parse_form4(
        self, cik: str, accession: str, document: str,
        fallback_ticker: str = "",
    ) -> list[dict]:
        self._rate_limit()
        url = f"{SEC_ARCHIVES}/{cik}/{accession}/{document}"
        resp = httpx.get(url, headers=self._headers, follow_redirects=True)
        if resp.status_code != 200:
            return []
        try:
            root = ElementTree.fromstring(resp.text)
        except ElementTree.ParseError:
            return []

        ticker = root.findtext(".//issuerTradingSymbol", "").strip().upper()
        if not ticker:
            ticker = fallback_ticker
        owner_name = root.findtext(".//rptOwnerName", "")
        title = root.findtext(".//officerTitle", "")

        transactions = []
        for txn in root.findall(".//nonDerivativeTransaction"):
            code = txn.findtext(".//transactionCode", "")
            if code != "P":
                continue
            txn_date = txn.findtext(".//transactionDate/value", "")
            shares_str = txn.findtext(
                ".//transactionAmounts/transactionShares/value", "0"
            )
            price_str = txn.findtext(
                ".//transactionAmounts/transactionPricePerShare/value", "0"
            )
            try:
                shares = float(shares_str)
                price = float(price_str) if price_str else 0.0
            except ValueError:
                continue
            if shares <= 0 or price <= 0:
                continue
            transactions.append({
                "ticker": ticker,
                "date": txn_date,
                "insider": owner_name,
                "title": title,
                "shares": shares,
                "price": price,
                "value": round(shares * price, 2),
            })
        return transactions

    def get_insider_purchases(
        self,
        tickers: list[str],
        lookback_days: int = 90,
    ) -> pl.DataFrame:
        after = date.today() - timedelta(days=lookback_days)
        all_txns: list[dict] = []

        for ticker in tickers:
            cik = self._get_cik(ticker)
            if not cik:
                logger.debug("No CIK found for %s", ticker)
                continue
            filings = self._get_form4_accessions(cik, after)
            logger.info("%s: %d Form 4 filings since %s", ticker, len(filings), after)
            for filing in filings:
                txns = self._parse_form4(
                    cik, filing["accession"], filing["document"],
                    fallback_ticker=ticker.upper(),
                )
                all_txns.extend(txns)

        if not all_txns:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "date": pl.Utf8,
                    "insider": pl.Utf8,
                    "title": pl.Utf8,
                    "shares": pl.Float64,
                    "price": pl.Float64,
                    "value": pl.Float64,
                }
            )
        return pl.DataFrame(all_txns)

    def get_purchase_summary(
        self,
        tickers: list[str],
        lookback_days: int = 90,
    ) -> pl.DataFrame:
        """Aggregate insider purchases per ticker."""
        txns = self.get_insider_purchases(tickers, lookback_days)
        if txns.is_empty():
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "num_buyers": pl.UInt32,
                    "total_shares": pl.Float64,
                    "total_value": pl.Float64,
                    "num_purchases": pl.UInt32,
                }
            )
        return (
            txns.group_by("ticker")
            .agg(
                pl.col("insider").n_unique().alias("num_buyers"),
                pl.col("shares").sum().alias("total_shares"),
                pl.col("value").sum().alias("total_value"),
                pl.len().alias("num_purchases"),
            )
            .sort("total_value", descending=True)
        )
