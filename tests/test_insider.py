from __future__ import annotations

from xml.etree import ElementTree

import polars as pl
from screener.data.insider import InsiderProvider

FORM4_PURCHASE = """\
<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerTradingSymbol>ACME</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>DOE JANE</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <officerTitle>CFO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-03-15</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

FORM4_GRANT_AND_SALE = """\
<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerTradingSymbol>ACME</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>SMITH BOB</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-03-15</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-03-16</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>2000</value></transactionShares>
        <transactionPricePerShare><value>55.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-03-16</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>3000</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-03-16</value></transactionDate>
      <transactionCoding><transactionCode>F</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>800</value></transactionShares>
        <transactionPricePerShare><value>55.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

FORM4_EMPTY_TICKER = """\
<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerTradingSymbol>  </issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>JONES ALICE</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <officerTitle></officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-04-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>25.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

FORM4_ZERO_PRICE_PURCHASE = """\
<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerTradingSymbol>ACME</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>LEE CHARLIE</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <officerTitle></officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-04-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


class FakeInsiderProvider(InsiderProvider):
    """InsiderProvider with HTTP calls stubbed out."""

    def __init__(self, xml_responses: list[str]):
        super().__init__("test agent")
        self._xml_responses = xml_responses

    def _rate_limit(self) -> None:
        pass

    def _parse_form4(
        self, cik: str, accession: str, document: str,
        fallback_ticker: str = "",
    ) -> list[dict]:
        if not self._xml_responses:
            return []
        xml = self._xml_responses.pop(0)
        try:
            root = ElementTree.fromstring(xml)
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


def test_parse_purchase():
    # Parse directly from XML
    root = ElementTree.fromstring(FORM4_PURCHASE)
    ticker = root.findtext(".//issuerTradingSymbol", "").strip().upper()
    assert ticker == "ACME"

    txns = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code = txn.findtext(".//transactionCode", "")
        if code != "P":
            continue
        shares = float(txn.findtext(".//transactionAmounts/transactionShares/value", "0"))
        price = float(txn.findtext(".//transactionAmounts/transactionPricePerShare/value", "0"))
        txns.append({"shares": shares, "price": price})

    assert len(txns) == 1
    assert txns[0]["shares"] == 1000
    assert txns[0]["price"] == 50.00


def test_filters_non_purchases():
    root = ElementTree.fromstring(FORM4_GRANT_AND_SALE)
    codes = [
        txn.findtext(".//transactionCode", "")
        for txn in root.findall(".//nonDerivativeTransaction")
    ]
    assert codes == ["A", "S", "M", "F"]
    purchases = [c for c in codes if c == "P"]
    assert len(purchases) == 0


def test_empty_ticker_uses_fallback():
    root = ElementTree.fromstring(FORM4_EMPTY_TICKER)
    ticker = root.findtext(".//issuerTradingSymbol", "").strip().upper()
    assert ticker == ""

    provider = FakeInsiderProvider([FORM4_EMPTY_TICKER])
    txns = provider._parse_form4("123", "acc", "doc", fallback_ticker="XYZ")
    assert len(txns) == 1
    assert txns[0]["ticker"] == "XYZ"


def test_zero_price_purchase_filtered():
    root = ElementTree.fromstring(FORM4_ZERO_PRICE_PURCHASE)
    txn = root.find(".//nonDerivativeTransaction")
    code = txn.findtext(".//transactionCode", "")
    price = float(txn.findtext(".//transactionAmounts/transactionPricePerShare/value", "0"))
    assert code == "P"
    assert price == 0.0

    provider = FakeInsiderProvider([FORM4_ZERO_PRICE_PURCHASE])
    txns = provider._parse_form4("123", "acc", "doc")
    assert len(txns) == 0


def test_purchase_summary_aggregation():
    # Manually build transactions
    txns_list = [
        {"ticker": "ACME", "date": "2026-03-15", "insider": "DOE JANE",
         "title": "CFO", "shares": 1000.0, "price": 50.0, "value": 50000.0},
        {"ticker": "ACME", "date": "2026-03-16", "insider": "DOE JANE",
         "title": "CFO", "shares": 500.0, "price": 52.0, "value": 26000.0},
        {"ticker": "XYZ", "date": "2026-03-17", "insider": "SMITH BOB",
         "title": "CEO", "shares": 2000.0, "price": 30.0, "value": 60000.0},
    ]
    txns = pl.DataFrame(txns_list)

    summary = (
        txns.group_by("ticker")
        .agg(
            pl.col("insider").n_unique().alias("num_buyers"),
            pl.col("shares").sum().alias("total_shares"),
            pl.col("value").sum().alias("total_value"),
            pl.len().alias("num_purchases"),
        )
        .sort("total_value", descending=True)
    )

    assert len(summary) == 2
    acme = summary.filter(pl.col("ticker") == "ACME")
    assert acme["num_buyers"][0] == 1
    assert acme["total_shares"][0] == 1500.0
    assert acme["total_value"][0] == 76000.0
    assert acme["num_purchases"][0] == 2

    xyz = summary.filter(pl.col("ticker") == "XYZ")
    assert xyz["num_buyers"][0] == 1
    assert xyz["total_value"][0] == 60000.0


def test_xsl_prefix_stripping():
    """Verify XSLT prefixes are stripped from document paths."""
    for prefix in ("xslF345X03/", "xslF345X04/", "xslF345X05/", "xslF345X06/"):
        doc = f"{prefix}form4.xml"
        if doc.startswith("xslF345X") and "/" in doc:
            doc = doc.split("/", 1)[1]
        assert doc == "form4.xml"

    # No prefix — should be unchanged
    doc = "form4.xml"
    if doc.startswith("xslF345X") and "/" in doc:
        doc = doc.split("/", 1)[1]
    assert doc == "form4.xml"


def test_mixed_transactions_only_returns_purchases():
    provider = FakeInsiderProvider([FORM4_GRANT_AND_SALE])
    txns = provider._parse_form4("123", "acc", "doc")
    assert len(txns) == 0


def test_empty_results_schema():
    provider = InsiderProvider("test agent")
    empty = provider.get_purchase_summary([], lookback_days=90)
    assert "ticker" in empty.columns
    assert "num_buyers" in empty.columns
    assert "total_value" in empty.columns
    assert len(empty) == 0
