import logging
from unittest.mock import MagicMock

import pytest
from screener.trading.broker import RebalanceOrder
from screener.trading.etoro import EtoroBroker, EtoroPosition, compute_equity


@pytest.fixture(autouse=True)
def _no_settle_sleep(monkeypatch):
    """Zero the close/open verification settle so execute_orders tests stay fast."""
    monkeypatch.setattr("screener.trading.etoro._CLOSE_SETTLE_SECONDS", 0.0)


def test_equity_positions_only():
    """Simple case: credit + positions with PnL, no mirrors or orders."""
    portfolio = {
        "credit": 5000,
        "positions": [
            {"amount": 1000, "pnL": 50},
            {"amount": 2000, "pnL": -100},
        ],
        "mirrors": [],
        "ordersForOpen": [],
        "orders": [],
    }
    result = compute_equity(portfolio)

    # Available cash = 5000 - 0 - 0 = 5000
    # Total invested = 3000 (positions) + 0 + 0 = 3000
    # Unrealized PnL = 50 + (-100) = -50
    # Equity = 5000 + 3000 + (-50) = 7950
    assert result["cash"] == 5000
    assert result["equity"] == 7950
    assert result["portfolio_value"] == 2950  # 3000 + (-50)


def test_equity_with_pending_orders():
    """Pending orders reduce available cash and count as invested."""
    portfolio = {
        "credit": 10000,
        "positions": [{"amount": 2000, "pnL": 200}],
        "mirrors": [],
        "ordersForOpen": [
            {"amount": 500, "mirrorID": 0, "totalExternalCosts": 10},
        ],
        "orders": [{"amount": 300}],
    }
    result = compute_equity(portfolio)

    # Available cash = 10000 - 500 - 300 = 9200
    # Total invested = 2000 + 0 + 0 + 500 + 300 + 10 = 2810
    # Unrealized PnL = 200
    # Equity = 9200 + 2810 + 200 = 12210
    assert result["cash"] == 9200
    assert result["equity"] == 12210


def test_equity_with_mirrors():
    """Mirror (copy trading) positions contribute to invested and PnL."""
    portfolio = {
        "credit": 8000,
        "positions": [{"amount": 1000, "pnL": 100}],
        "mirrors": [
            {
                "availableAmount": 2000,
                "closedPositionsNetProfit": 300,
                "positions": [
                    {"amount": 500, "pnL": 75},
                ],
            },
        ],
        "ordersForOpen": [],
        "orders": [],
    }
    result = compute_equity(portfolio)

    # Available cash = 8000
    # Positions invested = 1000
    # Mirror positions invested = 500
    # Mirror adjusted = 2000 - 300 = 1700
    # Total invested = 1000 + 500 + 1700 = 3200
    # Positions PnL = 100
    # Mirror PnL = 75
    # Closed profit = 300
    # Unrealized PnL = 100 + 75 + 300 = 475
    # Equity = 8000 + 3200 + 475 = 11675
    assert result["cash"] == 8000
    assert result["equity"] == 11675
    assert result["portfolio_value"] == 3675  # 3200 + 475


def test_equity_mirror_orders_excluded_from_cash():
    """Orders with mirrorID != 0 should NOT reduce available cash."""
    portfolio = {
        "credit": 5000,
        "positions": [],
        "mirrors": [],
        "ordersForOpen": [
            {"amount": 1000, "mirrorID": 0, "totalExternalCosts": 0},
            {"amount": 2000, "mirrorID": 42, "totalExternalCosts": 0},
        ],
        "orders": [],
    }
    result = compute_equity(portfolio)

    # Only mirrorID=0 order reduces cash
    assert result["cash"] == 4000  # 5000 - 1000


def test_equity_empty_portfolio():
    """Brand new account with just credit."""
    portfolio = {
        "credit": 100000,
        "positions": [],
        "mirrors": [],
        "ordersForOpen": [],
        "orders": [],
    }
    result = compute_equity(portfolio)

    assert result["equity"] == 100000
    assert result["cash"] == 100000
    assert result["portfolio_value"] == 0


def test_equity_camelcase_field_variants():
    """API returns inconsistent casing (pnL vs pnl, mirrorID vs mirrorId)."""
    portfolio = {
        "credit": 3000,
        "positions": [{"amount": 1000, "pnl": 50}],
        "mirrors": [],
        "ordersForOpen": [
            {"amount": 200, "mirrorId": 0, "totalExternalCosts": 5},
        ],
        "orders": [],
    }
    result = compute_equity(portfolio)

    assert result["cash"] == 2800  # 3000 - 200
    # Total invested = 1000 + 200 + 5 = 1205
    # PnL = 50
    # Equity = 2800 + 1205 + 50 = 4055
    assert result["equity"] == 4055


def test_equity_nested_unrealized_pnl():
    """PnL endpoint nests PnL inside unrealizedPnL dict."""
    portfolio = {
        "credit": 500,
        "positions": [
            {"amount": 1000, "unrealizedPnL": {"pnL": 200}},
            {"amount": 2000, "unrealizedPnL": {"pnL": -50}},
        ],
        "mirrors": [],
        "ordersForOpen": [],
        "orders": [],
    }
    result = compute_equity(portfolio)

    # Unrealized PnL = 200 + (-50) = 150
    # Equity = 500 + 3000 + 150 = 3650
    assert result["equity"] == 3650
    assert result["portfolio_value"] == 3150


def test_equity_mixed_pnl_formats():
    """Some positions have nested unrealizedPnL, some have top-level pnL."""
    portfolio = {
        "credit": 1000,
        "positions": [
            {"amount": 500, "unrealizedPnL": {"pnL": 100}},
            {"amount": 500, "pnL": 50},
        ],
        "mirrors": [],
        "ordersForOpen": [],
        "orders": [],
    }
    result = compute_equity(portfolio)

    # PnL = 100 + 50 = 150
    assert result["equity"] == 2150


def test_equity_no_pnl_field():
    """Position with neither unrealizedPnL nor pnL defaults to 0."""
    portfolio = {
        "credit": 1000,
        "positions": [{"amount": 500}],
        "mirrors": [],
        "ordersForOpen": [],
        "orders": [],
    }
    result = compute_equity(portfolio)
    assert result["equity"] == 1500


class TestExecuteTrim:
    """Partial close trims overweight positions by the right number of units."""

    def _make_broker(self):
        broker = EtoroBroker(api_key="k", user_key="u", demo=True)
        broker._close_position = MagicMock(return_value={"orderForClose": {}})
        return broker

    def test_trim_deducts_correct_units(self):
        from screener.trading.broker import RebalanceOrder
        from screener.trading.etoro import EtoroPosition

        broker = self._make_broker()
        order = RebalanceOrder(ticker="MU", side="sell", notional=500, trim=True)
        positions = [
            EtoroPosition(position_id=1, instrument_id=1130, ticker="MU",
                         amount=2000, units=10.0, open_rate=200.0, pnl=500),
        ]

        broker._execute_trim(order, positions)

        broker._close_position.assert_called_once()
        call_args = broker._close_position.call_args
        assert call_args[0][0] == 1  # position_id
        assert call_args[0][1] == 1130  # instrument_id
        # $500 / ($2500 / 10 units) = 2.0 units
        assert abs(call_args[1]["units_to_deduct"] - 2.0) < 0.01

    def test_trim_caps_at_position_units(self):
        from screener.trading.broker import RebalanceOrder
        from screener.trading.etoro import EtoroPosition

        broker = self._make_broker()
        order = RebalanceOrder(ticker="MU", side="sell", notional=10000, trim=True)
        positions = [
            EtoroPosition(position_id=1, instrument_id=1130, ticker="MU",
                         amount=1000, units=5.0, open_rate=200.0, pnl=0),
        ]

        broker._execute_trim(order, positions)

        call_args = broker._close_position.call_args
        # Can't sell more than 5 * 0.999 = 4.995 units
        assert call_args[1]["units_to_deduct"] <= 5.0

    def test_trim_spreads_across_positions(self):
        from screener.trading.broker import RebalanceOrder
        from screener.trading.etoro import EtoroPosition

        broker = self._make_broker()
        order = RebalanceOrder(ticker="MU", side="sell", notional=600, trim=True)
        positions = [
            EtoroPosition(position_id=1, instrument_id=1130, ticker="MU",
                         amount=500, units=2.5, open_rate=200.0, pnl=0),
            EtoroPosition(position_id=2, instrument_id=1130, ticker="MU",
                         amount=500, units=2.5, open_rate=200.0, pnl=0),
        ]

        broker._execute_trim(order, positions)

        # $600 / ($1000 / 5 units) = 3.0 units total
        # Should deduct from largest first (both equal at 2.5)
        assert broker._close_position.call_count == 2


class TestExecuteOrdersStopLoss:
    """Stop-loss must be set on every buy or the buy is skipped."""

    def _make_broker(self):
        broker = EtoroBroker(api_key="k", user_key="u", demo=True)
        broker._instrument_cache = {"AAPL": 1129, "MSFT": 1130}
        return broker

    def test_buys_clamped_to_available_cash(self):
        """eToro Phase-2 buys must not exceed available cash even if the equity-
        based target is larger (just-closed cash may not have settled)."""
        from screener.trading.broker import RebalanceOrder

        broker = self._make_broker()
        broker.get_account = MagicMock(return_value={
            "equity": 100000, "cash": 100, "buying_power": 100, "portfolio_value": 100,
        })
        broker.cancel_open_orders = MagicMock(return_value=0)
        broker.get_positions_detailed = MagicMock(return_value=[])
        broker.resolve_positions = MagicMock(return_value={})
        broker._get_rate = MagicMock(return_value=None)
        broker._open_position = MagicMock(return_value={"orderForOpen": {}})

        # 1 target → equal-weight target = full $100k equity, but cash is only $100
        orders = [RebalanceOrder(ticker="AAPL", side="buy", notional=1)]
        broker.execute_orders(orders, dry_run=False, stop_loss_pct=0.0)

        notional = broker._open_position.call_args[0][1]  # (iid, notional, ...)
        assert notional <= 100

    def test_buy_skipped_when_rate_unavailable(self):
        from screener.trading.broker import RebalanceOrder

        broker = self._make_broker()
        broker.get_account = MagicMock(return_value={
            "equity": 10000, "cash": 5000, "buying_power": 5000, "portfolio_value": 5000,
        })
        broker.get_positions = MagicMock(return_value={})
        broker.cancel_open_orders = MagicMock(return_value=0)
        broker.get_positions_detailed = MagicMock(return_value=[])
        broker._get_rate = MagicMock(return_value=None)

        orders = [RebalanceOrder(ticker="AAPL", side="buy", notional=1000)]
        results = broker.execute_orders(orders, dry_run=False, stop_loss_pct=0.15)

        assert results[0].status == "error: could not get rate for stop-loss"

    def test_buy_proceeds_when_rate_available(self):
        from screener.trading.broker import RebalanceOrder

        broker = self._make_broker()
        broker.get_account = MagicMock(return_value={
            "equity": 10000, "cash": 5000, "buying_power": 5000, "portfolio_value": 5000,
        })
        broker.get_positions = MagicMock(return_value={})
        broker.resolve_positions = MagicMock(return_value={})
        broker.cancel_open_orders = MagicMock(return_value=0)
        # Phase-1 read (nothing to sell), then the buy-verify re-read shows the
        # opened AAPL position as held → stays "submitted".
        broker.get_positions_detailed = MagicMock(side_effect=[
            [],
            [EtoroPosition(position_id=1, instrument_id=1129, ticker="AAPL",
                           amount=1000, units=5.0, open_rate=200.0, pnl=0)],
        ])
        broker._get_rate = MagicMock(return_value=150.0)
        broker._open_position = MagicMock(return_value={"orderForOpen": {}})

        orders = [RebalanceOrder(ticker="AAPL", side="buy", notional=1000)]
        results = broker.execute_orders(orders, dry_run=False, stop_loss_pct=0.15)

        assert results[0].status == "submitted"
        call_args = broker._open_position.call_args
        assert call_args[0][0] == 1129
        assert call_args[1]["stop_loss_rate"] == 150.0 * 0.85

    def test_buy_no_stop_loss_when_pct_zero(self):
        from screener.trading.broker import RebalanceOrder

        broker = self._make_broker()
        broker.get_account = MagicMock(return_value={
            "equity": 10000, "cash": 5000, "buying_power": 5000, "portfolio_value": 5000,
        })
        broker.get_positions = MagicMock(return_value={})
        broker.resolve_positions = MagicMock(return_value={})
        broker.cancel_open_orders = MagicMock(return_value=0)
        broker.get_positions_detailed = MagicMock(side_effect=[
            [],
            [EtoroPosition(position_id=1, instrument_id=1129, ticker="AAPL",
                           amount=1000, units=5.0, open_rate=200.0, pnl=0)],
        ])
        broker._open_position = MagicMock(return_value={"orderForOpen": {}})

        orders = [RebalanceOrder(ticker="AAPL", side="buy", notional=1000)]
        results = broker.execute_orders(orders, dry_run=False, stop_loss_pct=0.0)

        assert results[0].status == "submitted"
        call_args = broker._open_position.call_args
        assert call_args[0][0] == 1129
        assert call_args[1]["stop_loss_rate"] is None


class TestExecuteOrdersUnclearedExit:
    """A full-exit close that eToro silently no-ops (returns HTTP 200 under PDT
    rules) leaves the name still held on re-query → the broker must mark the sell
    an error and abort the buys, mirroring AlpacaBroker's post-sell verification.
    """

    def _make_broker(self, monkeypatch):
        monkeypatch.setattr("screener.trading.etoro._CLOSE_SETTLE_SECONDS", 0.0)
        broker = EtoroBroker(api_key="k", user_key="u", demo=True)
        broker._instrument_cache = {"OLD": 1111, "NEW": 2222}
        broker.cancel_open_orders = MagicMock(return_value=0)
        broker._close_position = MagicMock(return_value={"orderForClose": {}})  # 200, no raise
        broker.get_account = MagicMock(return_value={
            "equity": 5000, "cash": 5000, "buying_power": 5000, "portfolio_value": 5000})
        broker.resolve_positions = MagicMock(return_value={})
        broker._open_position = MagicMock(return_value={"orderForOpen": {}})
        return broker

    @staticmethod
    def _old_pos():
        return EtoroPosition(position_id=10, instrument_id=1111, ticker="OLD",
                             amount=5000, units=10.0, open_rate=500.0, pnl=0)

    def test_uncleared_full_exit_aborts_buys(self, monkeypatch):
        broker = self._make_broker(monkeypatch)
        # Still held on BOTH the Phase-1 read and the verification re-read:
        # the close returned 200 but silently did nothing.
        broker.get_positions_detailed = MagicMock(return_value=[self._old_pos()])
        orders = [
            RebalanceOrder(ticker="OLD", side="sell", notional=5000),  # full exit
            RebalanceOrder(ticker="NEW", side="buy", notional=5000),
        ]
        result = broker.execute_orders(orders, dry_run=False)
        status = {o.ticker: o.status for o in result}
        assert status["OLD"].startswith("error")
        assert status["NEW"] == "aborted"
        broker._open_position.assert_not_called()

    def test_cleared_full_exit_allows_buys(self, monkeypatch):
        broker = self._make_broker(monkeypatch)
        new_pos = EtoroPosition(position_id=20, instrument_id=2222, ticker="NEW",
                                amount=5000, units=10.0, open_rate=500.0, pnl=0)
        # Reads in order: Phase-1 (OLD held) → close-verify (OLD gone, cleared)
        # → buy-verify (NEW now held, open cleared).
        broker.get_positions_detailed = MagicMock(
            side_effect=[[self._old_pos()], [], [new_pos]])
        orders = [
            RebalanceOrder(ticker="OLD", side="sell", notional=5000),
            RebalanceOrder(ticker="NEW", side="buy", notional=5000),
        ]
        result = broker.execute_orders(orders, dry_run=False)
        status = {o.ticker: o.status for o in result}
        assert status["OLD"] == "submitted"
        assert status["NEW"] == "submitted"
        broker._open_position.assert_called_once()

    def test_trim_residual_not_flagged_as_failed_exit(self, monkeypatch):
        """A trim leaves a residual position by design — it must NOT be treated
        as a failed close even though the ticker is still held afterward."""
        broker = self._make_broker(monkeypatch)
        broker._execute_trim = MagicMock()
        broker.get_positions_detailed = MagicMock(return_value=[self._old_pos()])
        orders = [RebalanceOrder(ticker="OLD", side="sell", notional=1000, trim=True)]
        result = broker.execute_orders(orders, dry_run=False)
        assert result[0].status == "submitted"  # residual is expected, not a failure


class TestExecuteOrdersWeightCompleteness:
    """When target_weights is provided, a buy missing from it silently falls back
    to equal-weight sizing with a different denominator than the preview — warn."""

    def _make_broker(self, monkeypatch):
        monkeypatch.setattr("screener.trading.etoro._CLOSE_SETTLE_SECONDS", 0.0)
        broker = EtoroBroker(api_key="k", user_key="u", demo=True)
        broker._instrument_cache = {"NEW": 2222}
        broker.cancel_open_orders = MagicMock(return_value=0)
        broker.get_positions_detailed = MagicMock(return_value=[])
        broker.resolve_positions = MagicMock(return_value={})
        broker.get_account = MagicMock(return_value={
            "equity": 5000, "cash": 5000, "buying_power": 5000, "portfolio_value": 5000})
        broker._open_position = MagicMock(return_value={"orderForOpen": {}})
        return broker

    def test_missing_target_weight_warns(self, monkeypatch, caplog):
        broker = self._make_broker(monkeypatch)
        orders = [RebalanceOrder(ticker="NEW", side="buy", notional=1000)]
        with caplog.at_level(logging.WARNING):
            broker.execute_orders(orders, dry_run=False, target_weights={"OTHER": 1.0})
        assert any("missing from target_weights" in r.getMessage() for r in caplog.records)


class TestExecuteOrdersUnclearedBuy:
    """_open_position returns HTTP 200 even on a silent reject, so an open that
    produced no holding must be surfaced as an error, not logged as submitted."""

    def _make_broker(self):
        broker = EtoroBroker(api_key="k", user_key="u", demo=True)
        broker._instrument_cache = {"NEW": 2222}
        broker.cancel_open_orders = MagicMock(return_value=0)
        broker.resolve_positions = MagicMock(return_value={})
        broker.get_account = MagicMock(return_value={
            "equity": 5000, "cash": 5000, "buying_power": 5000, "portfolio_value": 5000})
        broker._open_position = MagicMock(return_value={"orderForOpen": {}})
        return broker

    def test_uncleared_open_marked_error(self):
        broker = self._make_broker()
        # Phase-1 read: no positions; buy-verify re-read: NEW still absent → failed.
        broker.get_positions_detailed = MagicMock(side_effect=[[], []])
        orders = [RebalanceOrder(ticker="NEW", side="buy", notional=5000)]
        result = broker.execute_orders(orders, dry_run=False)
        assert result[0].status.startswith("error")
        broker._open_position.assert_called_once()  # the open WAS attempted

    def test_cleared_open_stays_submitted(self):
        broker = self._make_broker()
        new_pos = EtoroPosition(position_id=9, instrument_id=2222, ticker="NEW",
                                amount=5000, units=10.0, open_rate=500.0, pnl=0)
        broker.get_positions_detailed = MagicMock(side_effect=[[], [new_pos]])
        orders = [RebalanceOrder(ticker="NEW", side="buy", notional=5000)]
        result = broker.execute_orders(orders, dry_run=False)
        assert result[0].status == "submitted"


class TestInstrumentCacheCollision:
    """A held non-US instrument must not poison the forward buy-target cache."""

    def test_reverse_lookup_does_not_clobber_forward_stock_mapping(self):
        # Seagate stock is "STX.US" (iid 8543); Stacks crypto is bare "STX" (iid 999).
        broker = EtoroBroker(api_key="k", user_key="u", demo=True)

        def fake_search(**params):
            if params.get("internalSymbolFull") == "STX":
                # Forward search returns both; resolve_instrument_id must prefer .US
                return [
                    {"internalSymbolFull": "STX", "instrumentId": 999},
                    {"internalSymbolFull": "STX.US", "instrumentId": 8543},
                ]
            if params.get("instrumentId") == 999:
                return [{"internalSymbolFull": "STX", "instrumentId": 999}]
            return []

        broker._search_instruments = fake_search
        # Account holds the crypto (bare "STX", iid 999) as an unresolved instrument.
        broker.get_positions_detailed = MagicMock(return_value=[
            EtoroPosition(position_id=1, instrument_id=999, ticker="ID:999",
                          amount=1000, units=1.0, open_rate=1000.0, pnl=0),
        ])

        # Forward-resolve the pick, then reverse-resolve the held crypto.
        broker.resolve_positions(candidate_tickers=["STX"])

        # The buy target for "STX" must remain Seagate stock, not the crypto.
        assert broker.resolve_instrument_id("STX") == 8543
        # Reverse cache is still populated so the held position reports as STX.
        assert broker._ticker_for_instrument(999) == "STX"


class TestAwaitPositionState:
    """eToro's positions endpoint lags fills, so verification POLLS rather than
    re-reading once — a real fill that shows up on a later poll must be confirmed,
    not false-failed on the first miss (the bug that aborted a valid rebalance)."""

    @staticmethod
    def _pos(ticker):
        return EtoroPosition(position_id=1, instrument_id=1, ticker=ticker,
                             amount=1, units=1.0, open_rate=1.0, pnl=0)

    def test_open_confirmed_on_a_later_poll(self):
        broker = EtoroBroker(api_key="k", user_key="u", demo=True)
        # Endpoint lags: empty on the first two reads, position appears on the 3rd.
        broker.get_positions_detailed = MagicMock(
            side_effect=[[], [], [self._pos("NEW")]])
        pending = broker._await_position_state(
            {"NEW"}, want_present=True, timeout=10, interval=0.001)
        assert pending == set()  # confirmed present
        assert broker.get_positions_detailed.call_count == 3

    def test_close_confirmed_on_a_later_poll(self):
        broker = EtoroBroker(api_key="k", user_key="u", demo=True)
        broker.get_positions_detailed = MagicMock(
            side_effect=[[self._pos("OLD")], [self._pos("OLD")], []])
        pending = broker._await_position_state(
            {"OLD"}, want_present=False, timeout=10, interval=0.001)
        assert pending == set()  # confirmed gone

    def test_times_out_when_fill_never_reflects(self):
        broker = EtoroBroker(api_key="k", user_key="u", demo=True)
        broker.get_positions_detailed = MagicMock(return_value=[])
        pending = broker._await_position_state(
            {"NEW"}, want_present=True, timeout=0.03, interval=0.01)
        assert pending == {"NEW"}  # never appeared → surfaced as unconfirmed

    def test_zero_interval_collapses_to_single_read(self):
        # Tests patch _CLOSE_SETTLE_SECONDS to 0 → exactly one re-read, so the
        # side_effect call counts other tests assert on stay correct.
        broker = EtoroBroker(api_key="k", user_key="u", demo=True)
        broker.get_positions_detailed = MagicMock(return_value=[])
        pending = broker._await_position_state(
            {"NEW"}, want_present=True, timeout=10, interval=0.0)
        assert pending == {"NEW"}
        assert broker.get_positions_detailed.call_count == 1
