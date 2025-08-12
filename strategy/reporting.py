from strategy.params import initialize_parameters

PARAMS = initialize_parameters()

# ==================== GENERATE SUMMARY REPORT ====================
def print_summary_report(issue_tracker, all_contracts):
    if not PARAMS.get('silent_mode', False):
        print("\n" + "=" * 20 + " SUMMARY OF ERRORS + WARNINGS " + "=" * 20)

        # Processing Stats Section
        print("\n📊 PROCESSING STATS:")
        print(f"  - Days attempted: {issue_tracker['days']['attempted']}")
        print(f"  - Days successfully processed: {issue_tracker['days']['processed']}")
        print(f"  - Days skipped due to errors: {issue_tracker['days']['skipped_errors']}")
        print(f"  - Days skipped due to warnings: {issue_tracker['days']['skipped_warnings']}")

        # Data Integrity Section
        print("\n🔍 DATA INTEGRITY:")
        print(f"  - Hash mismatches: {issue_tracker['data_integrity']['hash_mismatches']}")
        mismatch_count = issue_tracker['data_integrity']['timestamp_mismatches']
        days_with_mismatches = issue_tracker['data_integrity']['days_with_mismatches']
        if mismatch_count > 0:
            days_str = ", ".join(d for d in sorted(days_with_mismatches))
            print(f"  - Timestamp mismatches: {mismatch_count} (on {days_str})")
        else:
            print(f"  - Timestamp mismatches: {mismatch_count}")

        # Warning Summary
        print("\n⚠️ WARNING SUMMARY:")
        total_warnings = sum(count for key, count in issue_tracker['warnings'].items() if key != 'details')
        if total_warnings > 0:
            print("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
        ticker = PARAMS['ticker']
        print(f"  - No {ticker} Data: {issue_tracker['warnings'][f'no_{ticker}_data']}")
        print(f"  - Price staleness: {issue_tracker['warnings']['price_staleness']}")
        print(f"  - Short data warnings: {issue_tracker['warnings']['short_data_warnings']}")
        print(f"  - Timestamp mismatches below threshold: {issue_tracker['warnings']['timestamp_mismatches_below_threshold']}")
        print(f"  - Missing shares_per_contract: {issue_tracker['warnings']['shares_per_contract_missing']}")
        print(f"  - Non-standard contract size: {issue_tracker['warnings']['non_standard_contract_size']}")
        print(f"  - VWAP fallbacks to close price: {issue_tracker['warnings']['vwap_fallback_to_close']}")
        print(f"  - Emergency exit activations: {issue_tracker['warnings']['emergency_exit_triggered']}")
        print(f"  - Other warnings: {issue_tracker['warnings']['other']}")
        if issue_tracker['warnings']['other'] > 0 and issue_tracker['warnings']['details']:
            print("    Details:")
            for detail in issue_tracker['warnings']['details'][:5]:
                print(f"    - {detail}")
            if len(issue_tracker['warnings']['details']) > 5:
                print(f"    ... and {len(issue_tracker['warnings']['details']) - 5} more")

        # Error Summary
        print("\n❌ ERROR SUMMARY:")
        total_errors = sum(count for key, count in issue_tracker['errors'].items() if key != 'details')
        if total_errors > 0:
            print("❌❌❌❌❌❌❌❌❌❌")
        print(f"  - Missing option price data: {issue_tracker['errors']['missing_option_price_data']}")
        print(f"  - API connection failures: {issue_tracker['errors']['api_connection_failures']}")
        print(f"  - Missing exit data: {issue_tracker['errors']['missing_exit_data']}")
        print(f"  - No future price data: {issue_tracker['errors']['no_future_price_data']}")
        print(f"  - Forced exit at end of data: {issue_tracker['errors']['forced_exit_end_of_data']}")
        print(f"  - Exit evaluation error: {issue_tracker['errors']['exit_evaluation_error']}")
        print(f"  - Forced exit error: {issue_tracker['errors']['forced_exit_error']}")
        print(f"  - Latency entry failures: {issue_tracker['errors']['latency_entry_failures']}")
        print(f"  - Latency exit failures: {issue_tracker['errors']['latency_exit_failures']}")
        print(f"  - Other errors: {issue_tracker['errors']['other']}")
        if issue_tracker['errors']['other'] > 0 and issue_tracker['errors']['details']:
            print("    Details:")
            for detail in issue_tracker['errors']['details'][:5]:
                print(f"    - {detail}")
            if len(issue_tracker['errors']['details']) > 5:
                print(f"    ... and {len(issue_tracker['errors']['details']) - 5} more")

        # Opportunity Analysis
        print("\n🎯 OPPORTUNITY ANALYSIS:")
        print(f"  - Total stretch signals: {issue_tracker['opportunities']['total_stretch_signals']}")
        print(f"  - Valid entry opportunities: {issue_tracker['opportunities']['valid_entry_opportunities']}")
        print(f"  - Failed entries due to data issues: {issue_tracker['opportunities']['failed_entries_data_issues']}")
        print(f"  - Total options contracts selected: {issue_tracker['opportunities']['total_options_contracts']}")

        # Risk Management Statistics
        print("\n🛡️ RISK MANAGEMENT STATISTICS:")
        print(f"  - Emergency exits triggered: {issue_tracker['risk_management']['emergency_exits']}")
        if issue_tracker['risk_management']['emergency_exits'] > 0:
            emergency_dates = sorted(issue_tracker['risk_management']['emergency_exit_dates'])
            dates_str = ", ".join(emergency_dates)
            print(f"  - Dates with emergency exits: {dates_str}")
        print(f"  - Late entries blocked: {issue_tracker['risk_management']['late_entries_blocked']}")
        if issue_tracker['risk_management']['late_entries_blocked'] > 0:
            late_entry_dates = sorted(issue_tracker['risk_management']['late_entry_dates'])
            dates_str = ", ".join(late_entry_dates)
            print(f"  - Dates with blocked late entries: {dates_str}")
        print(f"  - Regular end-of-day exits: {sum(1 for c in all_contracts if c.get('exit_reason') == 'end_of_day')}")
        print(f"  - Total trades with defined exit reason: {sum(1 for c in all_contracts if c.get('exit_reason') is not None)}")
        print("\n" + "=" * 20 + " END OF REPORT " + "=" * 20)

# ==================== PERFORMANCE SUMMARY ====================
def print_performance_summary(contracts_df):
    """
    Print the performance summary given a DataFrame of contracts
    that already includes slippage and fee-adjusted columns.
    """
    if not PARAMS.get('silent_mode', False):
        if not contracts_df.empty:
            print("\n" + "=" * 20 + " PERFORMANCE SUMMARY " + "=" * 20)

            # Total number of trades
            total_trades = len(contracts_df)
            print(f"\n💼 TOTAL TRADES: {total_trades}")
            # Filter for trades with valid fully-adjusted P&L data (slippage + fees)
            valid_pnl_contracts = contracts_df.dropna(subset=['pnl_dollars_slipped_with_fees'])

            if not valid_pnl_contracts.empty:
                winning_trades = valid_pnl_contracts[valid_pnl_contracts['pnl_dollars_slipped_with_fees'] > 0]
                win_rate = len(winning_trades) / len(valid_pnl_contracts) * 100
                print(f"\n🎯 WIN RATE: {win_rate:.2f}%")

                # Expectancy
                if not winning_trades.empty:
                    avg_win = winning_trades['pnl_dollars_slipped_with_fees'].mean()
                    losing_trades = valid_pnl_contracts[valid_pnl_contracts['pnl_dollars_slipped_with_fees'] < 0]

                    if not losing_trades.empty:
                        avg_loss = abs(losing_trades['pnl_dollars_slipped_with_fees'].mean())
                        loss_rate = 1 - (len(winning_trades) / len(valid_pnl_contracts))
                        expectancy = (win_rate/100 * avg_win) - (loss_rate * avg_loss)
                        print(f"\n💡 EXPECTANCY: ${expectancy:.2f} per trade")
                    else:
                        print("\n💡 EXPECTANCY: ∞ (no losing trades)")

                # Average risk per trade
                contract_fees = PARAMS['brokerage_fee_per_contract'] + PARAMS['exchange_fee_per_contract']
                round_trip_fees = contract_fees * 2 * PARAMS['contracts_per_trade']
                stop_loss_decimal = abs(PARAMS['stop_loss_percent']) / 100
                valid_pnl_contracts['capital_at_risk'] = (
                    valid_pnl_contracts['entry_option_price_slipped'] * valid_pnl_contracts['shares_per_contract'] * PARAMS['contracts_per_trade']
                )
                valid_pnl_contracts['max_loss_per_trade'] = (valid_pnl_contracts['capital_at_risk'] * stop_loss_decimal) + round_trip_fees
                avg_risk = valid_pnl_contracts['max_loss_per_trade'].mean()
                print(f"\n💵 AVERAGE RISK PER TRADE: ${avg_risk:.2f}")

                # Risk-adjusted expectancy
                if 'expectancy' in locals() and avg_risk > 0:
                    if expectancy != float('inf'):
                        return_on_risk = expectancy / avg_risk * 100
                        print(f"\n📊 AVERAGE RETURN ON RISK: {return_on_risk:.2f}%")
                    else:
                        print("\n📊 AVERAGE RETURN ON RISK: ∞% (no losing trades)")

                # Sharpe Ratio
                daily_returns = valid_pnl_contracts.groupby(valid_pnl_contracts['entry_time'].dt.date)['pnl_dollars_slipped_with_fees'].sum()
                if len(daily_returns) > 1:
                    mean_daily = daily_returns.mean()
                    std_daily = daily_returns.std()
                    if std_daily > 0:
                        sharpe = mean_daily / std_daily
                        print(f"\n📈 UNANNUALIZED SHARPE RATIO: {sharpe:.2f}")
                    else:
                        print("\n📈 UNANNUALIZED SHARPE RATIO: N/A (insufficient volatility)")

                else:
                    print("\n📈 SHARPE RATIO: N/A (need data from at least two days)")

            print("\n" + "=" * 20 + " END OF PERFORMANCE SUMMARY " + "=" * 20)