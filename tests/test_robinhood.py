from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from screener.trading.robinhood import (
    LotPlan,
    RobinhoodBroker,
    RobinhoodPosition,
    eastern_now,
    market_is_open,
    parse_order_fills,
    plan_trim_lots,
)


def _lot(lot_id, qty, basis, selectable=True, **over):
    lot = {
        "open_lot_id": lot_id,
        "quantity": str(qty),
        "quantity_available": str(qty),
        "cost_per_share": str(basis),
        "is_selectable": selectable,
        "term": "st",
    }
    lot.update(over)
    return lot


class TestPlanTrimLots:

    def test_picks_highest_basis_first(self):
        # $300 trim at $100/share = 3 shares. Highest basis first: the 150 lot
        # (2 sh) then the 120 lot (1 of its 5) — never the cheap 50 lot.
        plan = plan_trim_lots(300.0, 100.0, [
            _lot("cheap", 10, 50), _lot("mid", 5, 120), _lot("dear", 2, 150),
        ])

        assert plan.tax_lots == [
            {"open_lot_id": "dear", "quantity": "2.000000"},
            {"open_lot_id": "mid", "quantity": "1.000000"},
        ]
        assert plan.quantity == "3.000000"

    def test_lot_quantities_sum_exactly_to_the_order_quantity(self):
        """The broker rejects a lot list that doesn't sum to the order quantity,
        so this must hold under awkward division, not just round numbers."""
        plan = plan_trim_lots(1000.0, 3.0, [       # 333.333333... shares
            _lot("a", 100, 90), _lot("b", 100, 80), _lot("c", 200, 70),
        ])

        total = sum(Decimal(lot["quantity"]) for lot in plan.tax_lots)
        assert total == Decimal(plan.quantity)
        assert plan.quantity == "333.333333"       # floored, never rounded up

    def test_never_oversells(self):
        plan = plan_trim_lots(100.0, 30.0, [_lot("a", 10, 90)])   # 3.33... shares
        assert Decimal(plan.quantity) <= Decimal("100") / Decimal("30")

    def test_skips_unselectable_lots(self):
        """A lot acquired the same session comes back is_selectable=false and
        cannot be named — picking it would have the whole order rejected."""
        plan = plan_trim_lots(100.0, 100.0, [
            _lot("today", 5, 999, selectable=False),   # highest basis, unusable
            _lot("older", 5, 100),
        ])

        assert [lot["open_lot_id"] for lot in plan.tax_lots] == ["older"]

    def test_derives_basis_when_cost_per_share_is_pending(self):
        lot = _lot("a", 4, 0, tax_cost_basis="800.00")
        del lot["cost_per_share"]
        plan = plan_trim_lots(100.0, 100.0, [lot, _lot("b", 4, 150)])

        # 800/4 = 200/share, above b's 150 → a is picked first.
        assert plan.tax_lots[0]["open_lot_id"] == "a"

    def test_respects_quantity_available(self):
        plan = plan_trim_lots(500.0, 100.0, [        # want 5 shares
            _lot("partly_sold", 10, 200, quantity_available="1"),
            _lot("other", 10, 100),
        ])

        assert plan.tax_lots[0] == {"open_lot_id": "partly_sold",
                                    "quantity": "1.000000"}
        assert plan.tax_lots[1]["quantity"] == "4.000000"

    def test_falls_back_to_fifo_when_lots_cannot_cover_the_trim(self):
        """Trim size is the primary goal, lot choice the secondary one — so an
        uncoverable trim still goes out at full size, as a plain FIFO sell."""
        plan = plan_trim_lots(1000.0, 100.0, [       # want 10 shares
            _lot("a", 2, 100), _lot("locked", 50, 300, selectable=False),
        ])

        assert plan.tax_lots == []                   # ⇒ omit tax_lots entirely
        assert plan.quantity == "10.000000"          # full size preserved
        assert "FIFO fallback" in plan.note and "still syncing" in plan.note

    def test_falls_back_when_more_than_the_lot_cap_is_needed(self):
        lots = [_lot(f"l{i}", 1, 100 + i) for i in range(40)]
        plan = plan_trim_lots(3500.0, 100.0, lots)   # 35 shares > 30 lots x 1

        assert plan.tax_lots == []
        assert "30-lot cap" in plan.note

    def test_uses_the_cap_when_it_is_just_enough(self):
        lots = [_lot(f"l{i}", 1, 100 + i) for i in range(40)]
        plan = plan_trim_lots(3000.0, 100.0, lots)   # exactly 30 shares

        assert len(plan.tax_lots) == 30
        assert sum(Decimal(x["quantity"]) for x in plan.tax_lots) == Decimal(30)

    def test_zero_and_bad_inputs_do_not_raise(self):
        assert plan_trim_lots(0.0, 100.0, []).tax_lots == []
        assert plan_trim_lots(100.0, 0.0, []).tax_lots == []
        assert plan_trim_lots(100.0, -5.0, []).tax_lots == []

    def test_sub_precision_trim_rounds_to_zero_shares(self):
        """Below the broker's 6dp share precision there is no order to place —
        report zero rather than rounding up into a trade nobody asked for."""
        plan = plan_trim_lots(0.0001, 1000.0, [_lot("a", 5, 100)])

        assert plan.quantity == "0"
        assert plan.tax_lots == []
        assert "zero shares" in plan.note

    def test_unread_pages_are_reported_not_hidden(self):
        """get_equity_tax_lots is paginated. Truncation both narrows the HIFO
        choice and can force a fallback — which would otherwise blame
        is_selectable — so an unread page has to show up in the note."""
        payload = {"data": {"tax_lots": [_lot("a", 1, 100)],
                            "next": "https://...?cursor=abc"}}

        covered = plan_trim_lots(100.0, 100.0, payload)          # 1 share, fits
        short = plan_trim_lots(500.0, 100.0, payload)            # 5 shares, no

        assert covered.tax_lots and "more lot pages" in covered.note
        assert short.tax_lots == [] and "more lot pages" in short.note

    def test_no_next_cursor_means_no_truncation_note(self):
        plan = plan_trim_lots(100.0, 100.0, {"data": {"tax_lots": [_lot("a", 5, 100)]}})
        assert "more lot pages" not in plan.note

    def test_real_sndk_payload(self):
        """The live get_equity_tax_lots response (aug-2026): 7 lots, the newest
        still syncing with no cost_per_share. A $2,000 trim at $1,429/share."""
        payload = {"data": {"symbol": "SNDK", "tax_lots": [
            {"open_lot_id": "d0731c58", "quantity": "0.555963",
             "quantity_available": "0.555963", "is_selectable": False,
             "tax_cost_basis": "796.250000", "open_date": "2026-08-04"},
            {"open_lot_id": "7b4bcdbf", "quantity": "0.471078",
             "quantity_available": "0.471078", "is_selectable": True,
             "cost_per_share": "1564.350000", "tax_cost_basis": "736.930000"},
            {"open_lot_id": "702adbc7", "quantity": "1.630301",
             "quantity_available": "1.630301", "is_selectable": True,
             "cost_per_share": "1711.340000", "tax_cost_basis": "2790.000000"},
            {"open_lot_id": "e0114093", "quantity": "0.683067",
             "quantity_available": "0.683067", "is_selectable": True,
             "cost_per_share": "2635.130000", "tax_cost_basis": "1799.970000"},
        ]}}

        plan = plan_trim_lots(2000.0, 1429.45, payload)

        # 1.399143 shares, taken from the dearest lots down: 2635 then 1711.
        assert [lot["open_lot_id"] for lot in plan.tax_lots] == ["e0114093",
                                                                "702adbc7"]
        assert sum(Decimal(x["quantity"]) for x in plan.tax_lots) \
            == Decimal(plan.quantity)
        assert "d0731c58" not in {lot["open_lot_id"] for lot in plan.tax_lots}


class TestParsePortfolio:

    def test_basic_portfolio(self):
        # Real get_portfolio MCP shape: nested buying_power, total_value = equity + cash
        data = {
            "total_value": "50000.0",
            "equity_value": "40000.0",
            "cash": "10000.0",
            "buying_power": {"buying_power": "10000.0"},
        }
        result = RobinhoodBroker.parse_portfolio(data)

        # Fully settled (buying_power == cash) → deployable base == total_value.
        assert result["equity"] == 50000.0
        assert result["cash"] == 10000.0
        assert result["buying_power"] == 10000.0
        assert result["portfolio_value"] == 40000.0
        assert result["unsettled_cash"] == 0.0

    def test_unwraps_data_envelope(self):
        data = {"data": {
            "total_value": "39847.40",
            "equity_value": "19838.78",
            "cash": "20008.62",
            "buying_power": {"buying_power": "20008.62"},
        }}
        result = RobinhoodBroker.parse_portfolio(data)

        assert result["equity"] == 39847.40
        assert result["cash"] == 20008.62
        assert result["buying_power"] == 20008.62

    def test_empty_portfolio(self):
        result = RobinhoodBroker.parse_portfolio({})

        assert result["equity"] == 0
        assert result["cash"] == 0
        assert result["buying_power"] == 0

    def test_margin_buying_power_does_not_inflate_the_sizing_base(self):
        # The agentic account is a MARGIN account: buying_power exceeds account
        # value by the margin multiple. The sizing base must stay the account's
        # own capital (total_value) — sizing to buying_power would lever the
        # book — and nothing may be treated as unsettled.
        data = {
            "total_value": "77315.88",
            "equity_value": "77266.42",
            "cash": "49.46",
            "buying_power": {"buying_power": "38000.00"},   # margin: ≫ cash
        }
        result = RobinhoodBroker.parse_portfolio(data)

        assert result["unsettled_cash"] == 0.0
        assert result["equity"] == 77315.88            # NOT total + margin
        assert result["buying_power"] == 38000.00      # reported, never sized off

    def test_unsettled_cash_excluded_from_sizing_base(self):
        # Cash-account safety net, retained after the aug-2026 margin upgrade:
        # $12,083 cash but only $5,176 tradable (~$6,907 sale proceeds still
        # settling). The sizing base must be the deployable capital (positions +
        # buying power), NOT total_value, so the equal-weight target isn't
        # inflated by cash that can't be traded yet.
        data = {
            "total_value": "86293.0",
            "equity_value": "74210.0",
            "cash": "12083.0",
            "buying_power": {"buying_power": "5176.0"},
        }
        result = RobinhoodBroker.parse_portfolio(data)

        assert result["unsettled_cash"] == 12083.0 - 5176.0      # 6907
        assert result["equity"] == 74210.0 + 5176.0              # 79386 deployable
        assert result["total_value"] == 86293.0                  # still exposed
        # deployable == total_value − unsettled_cash
        assert result["equity"] == result["total_value"] - result["unsettled_cash"]


class TestParsePositions:

    def test_with_quotes(self):
        positions = [
            {"symbol": "AAPL", "quantity": "10", "average_buy_price": "150.00"},
            {"symbol": "MSFT", "quantity": "5", "average_buy_price": "400.00"},
        ]
        quotes = {
            "AAPL": {"last_trade_price": "200.00"},
            "MSFT": {"last_trade_price": "420.00"},
        }

        result = RobinhoodBroker.parse_positions(positions, quotes)

        assert len(result) == 2
        assert result["AAPL"].market_value == 2000.0
        assert result["MSFT"].market_value == 2100.0
        assert result["AAPL"].quantity == 10.0

    def test_without_quotes_uses_avg_cost(self):
        positions = [
            {"symbol": "AAPL", "quantity": "10", "average_buy_price": "150.00"},
        ]

        result = RobinhoodBroker.parse_positions(positions)

        assert result["AAPL"].market_value == 1500.0

    def test_zero_quantity_excluded(self):
        positions = [
            {"symbol": "AAPL", "quantity": "0", "average_buy_price": "150.00"},
            {"symbol": "MSFT", "quantity": "5", "average_buy_price": "400.00"},
        ]

        result = RobinhoodBroker.parse_positions(positions)

        assert "AAPL" not in result
        assert "MSFT" in result

    def test_average_cost_field_variant(self):
        positions = [
            {"symbol": "AAPL", "quantity": "10", "average_cost": "150.00"},
        ]

        result = RobinhoodBroker.parse_positions(positions)

        assert result["AAPL"].average_cost == 150.0

    def test_extended_hours_price(self):
        positions = [
            {"symbol": "AAPL", "quantity": "10", "average_buy_price": "150.00"},
        ]
        quotes = {
            "AAPL": {"last_extended_hours_trade_price": "205.00"},
        }

        result = RobinhoodBroker.parse_positions(positions, quotes)

        assert result["AAPL"].market_value == 2050.0

    def test_null_last_trade_price_coalesces(self):
        """A present-but-null last_trade_price must fall back to ext-hours/avg_cost,
        not crash on float(None)."""
        positions = [
            {"symbol": "AAPL", "quantity": "10", "average_buy_price": "150.00"},
        ]
        quotes = {"AAPL": {"last_trade_price": None,
                           "last_extended_hours_trade_price": "205.00"}}

        result = RobinhoodBroker.parse_positions(positions, quotes)

        assert result["AAPL"].market_value == 2050.0  # used ext-hours, did not crash

    def test_zero_last_trade_price_coalesces_to_avg_cost(self):
        """A present-but-zero quote must not yield market_value 0 (phantom re-buy)."""
        positions = [
            {"symbol": "AAPL", "quantity": "10", "average_buy_price": "150.00"},
        ]
        quotes = {"AAPL": {"last_trade_price": "0.0000"}}

        result = RobinhoodBroker.parse_positions(positions, quotes)

        assert result["AAPL"].market_value == 1500.0  # avg_cost, not 0


class TestComputeRebalanceOrders:

    def _make_broker(self):
        return RobinhoodBroker(account_number="123456789")

    def test_buy_new_positions(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=["AAPL", "MSFT"],
            account={"equity": 10000},
            current={},
        )

        buys = [o for o in orders if o.side == "buy"]
        assert len(buys) == 2
        assert all(o.notional == 5000.0 for o in buys)

    def test_sell_removed_positions(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=["MSFT"],
            account={"equity": 10000},
            current={"AAPL": 5000, "MSFT": 5000},
        )

        sells = [o for o in orders if o.side == "sell"]
        assert len(sells) == 1
        assert sells[0].ticker == "AAPL"
        assert sells[0].notional == 5000

    def test_trim_overweight(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=["AAPL", "MSFT"],
            account={"equity": 10000},
            current={"AAPL": 8000, "MSFT": 2000},
        )

        sells = [o for o in orders if o.side == "sell"]
        buys = [o for o in orders if o.side == "buy"]
        assert len(sells) == 1
        assert sells[0].ticker == "AAPL"
        assert sells[0].trim is True
        assert sells[0].notional == 3000.0
        assert len(buys) == 1
        assert buys[0].ticker == "MSFT"

    def test_within_tolerance_no_orders(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=["AAPL", "MSFT"],
            account={"equity": 10000},
            current={"AAPL": 5100, "MSFT": 4900},
        )

        assert len(orders) == 0

    def test_empty_targets_sells_everything(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=[],
            account={"equity": 10000},
            current={"AAPL": 5000},
        )

        assert len(orders) == 1
        assert orders[0].side == "sell"
        assert orders[0].ticker == "AAPL"

    def test_no_positions_no_targets(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=[],
            account={"equity": 10000},
            current={},
        )

        assert len(orders) == 0

    def test_reserve_shrinks_sizing_base(self):
        # $10k deployable, reserve $4k → size to an $6k book (target $3k/name).
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=["AAPL", "MSFT"],
            account={"equity": 10000},
            current={},
            reserve=4000,
        )
        buys = [o for o in orders if o.side == "buy"]
        assert len(buys) == 2
        assert all(o.notional == 3000.0 for o in buys)  # (10000-4000)/2

    def test_reserve_above_cash_trims_to_smaller_book(self):
        # Fully invested $10k book; reserve $2k pushes the target book to $8k, so
        # each name trims from 5000 toward 4000 (valid "reduce exposure" input).
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=["AAPL", "MSFT"],
            account={"equity": 10000},
            current={"AAPL": 5000, "MSFT": 5000},
            reserve=2000,
        )
        sells = [o for o in orders if o.side == "sell"]
        assert len(sells) == 2
        assert all(o.trim and o.notional == 1000.0 for o in sells)  # 5000 - 8000/2

    def test_zero_reserve_matches_no_reserve(self):
        broker = self._make_broker()
        common = dict(target_tickers=["AAPL", "MSFT"], account={"equity": 10000},
                      current={"AAPL": 3000})
        a = broker.compute_rebalance_orders(**common)
        b = broker.compute_rebalance_orders(**common, reserve=0.0)
        assert [(o.ticker, o.side, o.notional) for o in a] \
            == [(o.ticker, o.side, o.notional) for o in b]


class TestFormatMcpOrder:

    def _make_broker(self):
        return RobinhoodBroker(account_number="123456789")

    def test_buy_uses_dollar_amount(self):
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="AAPL", side="buy", notional=1500.50)
        params = broker.format_mcp_order(order)

        assert params["account_number"] == "123456789"
        assert params["symbol"] == "AAPL"
        assert params["side"] == "buy"
        assert params["type"] == "market"
        assert params["dollar_amount"] == "1500.5"
        assert "quantity" not in params

    def test_full_exit_uses_quantity(self):
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="AAPL", side="sell", notional=3000)
        positions = {
            "AAPL": RobinhoodPosition(
                symbol="AAPL", quantity=15.5,
                average_cost=150, market_value=3000,
            ),
        }
        params = broker.format_mcp_order(order, positions=positions)

        assert params["quantity"] == "15.5"
        assert "dollar_amount" not in params

    def test_trim_uses_dollar_amount(self):
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(
            ticker="AAPL", side="sell", notional=500, trim=True,
        )
        positions = {
            "AAPL": RobinhoodPosition(
                symbol="AAPL", quantity=15.5,
                average_cost=150, market_value=3000,
            ),
        }
        params = broker.format_mcp_order(order, positions=positions)

        assert params["dollar_amount"] == "500"
        assert "quantity" not in params

    def test_trim_with_lot_plan_sells_by_quantity(self):
        """tax_lots is rejected alongside dollar_amount, so a lot-selected trim
        must switch to a quantity order."""
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="SNDK", side="sell", notional=2000,
                               trim=True)
        plan = LotPlan(quantity="1.399143",
                       tax_lots=[{"open_lot_id": "e01", "quantity": "1.399143"}],
                       note="HIFO across 1 lot(s)")
        params = broker.format_mcp_order(order, lot_plan=plan)

        assert params["quantity"] == "1.399143"
        assert params["tax_lots"] == plan.tax_lots
        assert "dollar_amount" not in params        # would be rejected together

    def test_trim_with_fifo_fallback_plan_stays_dollar_based(self):
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="SNDK", side="sell", notional=2000,
                               trim=True)
        plan = LotPlan(quantity="1.399143", tax_lots=[], note="FIFO fallback")
        params = broker.format_mcp_order(order, lot_plan=plan)

        assert params["dollar_amount"] == "2000"
        assert "tax_lots" not in params

    def test_trim_without_a_lot_plan_warns(self, caplog):
        """A forgotten planning step must not look like the deliberate FIFO
        fallback — nothing downstream can tell them apart afterwards."""
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="SNDK", side="sell", notional=2000,
                               trim=True)
        with caplog.at_level("WARNING"):
            params = broker.format_mcp_order(order)

        assert params["dollar_amount"] == "2000"
        assert "no lot plan" in caplog.text

    def test_deliberate_fifo_fallback_does_not_warn(self, caplog):
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="SNDK", side="sell", notional=2000,
                               trim=True)
        plan = LotPlan(quantity="1.4", tax_lots=[], note="FIFO fallback — …")
        with caplog.at_level("WARNING"):
            broker.format_mcp_order(order, lot_plan=plan)

        assert "no lot plan" not in caplog.text

    def test_position_price_is_the_sizing_price_not_a_quote_field(self):
        """Converting a trim's notional back to shares must use the price behind
        market_value; re-deriving from a quote diverges whenever parse_positions
        coalesced to ext-hours or average cost."""
        pos = RobinhoodPosition(symbol="SNDK", quantity=4.0,
                                average_cost=1000.0, market_value=5716.0)
        assert pos.price == 1429.0

    def test_position_price_of_empty_position_is_zero_not_a_crash(self):
        pos = RobinhoodPosition(symbol="X", quantity=0.0, average_cost=0.0,
                                market_value=0.0)
        assert pos.price == 0.0

    def test_full_exit_ignores_a_lot_plan(self):
        """A full exit realizes every lot, so naming lots adds only risk."""
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="AAPL", side="sell", notional=3000)
        positions = {"AAPL": RobinhoodPosition(
            symbol="AAPL", quantity=15.5, average_cost=150, market_value=3000)}
        plan = LotPlan(quantity="15.5",
                       tax_lots=[{"open_lot_id": "x", "quantity": "15.5"}],
                       note="")
        params = broker.format_mcp_order(order, positions, lot_plan=plan)

        assert params["quantity"] == "15.5"
        assert "tax_lots" not in params

    def test_buy_ignores_a_lot_plan(self):
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="AAPL", side="buy", notional=1500)
        plan = LotPlan(quantity="10",
                       tax_lots=[{"open_lot_id": "x", "quantity": "10"}], note="")
        params = broker.format_mcp_order(order, lot_plan=plan)

        assert params["dollar_amount"] == "1500"
        assert "tax_lots" not in params            # sell-only parameter

    def test_full_exit_no_positions_falls_back_to_dollar(self):
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="AAPL", side="sell", notional=3000)
        params = broker.format_mcp_order(order)

        assert params["dollar_amount"] == "3000"


class TestMode:

    def test_mode_is_live(self):
        broker = RobinhoodBroker(account_number="123")
        assert broker.mode == "LIVE"


class TestEasternNow:

    def test_utc_evening_is_already_next_day_in_utc_but_not_et(self):
        """The VM clock is UTC: 01:00 UTC Wednesday is still Tuesday in ET, and
        using the UTC date would misjudge the trading day."""
        now = datetime(2026, 8, 5, 1, 0, tzinfo=UTC)
        assert now.date() == date(2026, 8, 5)
        assert eastern_now(now).date() == date(2026, 8, 4)


class TestMarketIsOpen:

    def _quotes(self, ts):
        return {"A": {"venue_last_trade_time": ts}}

    def test_open_with_fresh_print(self):
        now = datetime(2026, 8, 4, 18, 13, 0, tzinfo=UTC)   # 14:13 ET, Tuesday
        ok, why = market_is_open(
            self._quotes("2026-08-04T18:12:37.923120425Z"), now)
        assert ok and "open" in why

    def test_weekend(self):
        now = datetime(2026, 8, 8, 18, 0, tzinfo=UTC)       # Saturday
        ok, why = market_is_open(
            self._quotes("2026-08-07T20:00:00.000000000Z"), now)
        assert not ok and "weekend" in why

    def test_outside_session_hours(self):
        now = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)       # 08:00 ET
        ok, why = market_is_open(
            self._quotes("2026-08-04T11:59:00.000000000Z"), now)
        assert not ok and "outside" in why

    def test_holiday_detected_by_stale_print(self):
        """Inside the ET window on a weekday, but nothing has printed — the
        empirical check catches a holiday with no calendar to maintain."""
        now = datetime(2026, 8, 4, 18, 0, tzinfo=UTC)
        ok, why = market_is_open(
            self._quotes("2026-08-03T20:00:00.000000000Z"), now)
        assert not ok and "market appears closed" in why

    def test_missing_timestamps_are_not_treated_as_open(self):
        now = datetime(2026, 8, 4, 18, 0, tzinfo=UTC)
        ok, why = market_is_open({"A": {}}, now)
        assert not ok and "no trade timestamps" in why


class TestParseOrderFills:

    def test_normalizes_mcp_order_response(self):
        rows = parse_order_fills({"data": {"orders": [{
            "symbol": "APP", "side": "buy", "state": "filled",
            "average_price": "415.799900", "cumulative_quantity": "4.075277",
            "fees": "0.000000",
            "dollar_based_amount": {"amount": "1694.500000"},
        }]}}, arrival={"APP": 415.79})

        assert rows == [{
            "ticker": "APP", "side": "buy", "notional": 1694.5,
            "status": "filled", "trim": False, "arrival_price": 415.79,
            "fill_price": 415.7999, "quantity": 4.075277, "fees": 0.0,
        }]

    def test_quantity_order_gets_a_notional(self):
        """Full exits are placed BY QUANTITY, so the MCP order has no
        dollar_based_amount. Without a derived notional,
        execution_decomposition skips the row (`notional <= 0` → continue) and
        the exec-quality table would measure only buys and trims."""
        rows = parse_order_fills([{
            "symbol": "RL", "side": "sell", "state": "filled",
            "average_price": "300.00", "cumulative_quantity": "10.5",
        }])

        assert rows[0]["notional"] == 3150.0        # 10.5 × 300, not None
        assert rows[0]["fill_price"] == 300.0

    def test_dollar_amount_wins_over_derived_notional(self):
        rows = parse_order_fills([{
            "symbol": "APP", "side": "buy", "state": "filled",
            "average_price": "100.00", "cumulative_quantity": "5",
            "dollar_based_amount": {"amount": "499.00"},
        }])

        assert rows[0]["notional"] == 499.0

    def test_unfilled_order_has_no_derived_notional(self):
        rows = parse_order_fills([{
            "symbol": "RL", "side": "sell", "state": "cancelled",
            "average_price": None, "cumulative_quantity": "0",
        }])

        assert rows[0]["notional"] is None          # nothing to invent

    def test_trims_are_labelled_from_the_computed_orders(self):
        """The MCP response can't distinguish a trim from a full exit, so the
        caller passes the trim set; a hardcoded False mislabels the log."""
        rows = parse_order_fills(
            [{"symbol": "STX", "side": "sell", "state": "filled",
              "average_price": "800.00", "cumulative_quantity": "1"},
             {"symbol": "RL", "side": "sell", "state": "filled",
              "average_price": "300.00", "cumulative_quantity": "10"},
             {"symbol": "STX", "side": "buy", "state": "filled",
              "average_price": "800.00", "cumulative_quantity": "1"}],
            trims={"STX"},
        )

        assert [r["trim"] for r in rows] == [True, False, False]
