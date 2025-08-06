import pytest
import pandas as pd
import numpy as np
from definition_f3f823e4507f4c2380669f8ba094b6a4 import compute_realized_lgd

def test_compute_realized_lgd_no_recoveries():
    # Mock data: EAD = 100, no recoveries or costs
    ead = pd.Series([100])
    recoveries_pv = pd.Series([0])
    collection_costs_pv = pd.Series([0])

    # Expected LGD: (100 - 0 - 0) / 100 = 1
    expected_lgd = pd.Series([1.0])

    # Monkeypatching is not ideal here as we don't have an object but it's kept to adhere to no dataset reading
    compute_realized_lgd.ead = ead
    compute_realized_lgd.recoveries_pv = recoveries_pv
    compute_realized_lgd.collection_costs_pv = collection_costs_pv

    lgd = compute_realized_lgd()
    pd.testing.assert_series_equal(lgd, expected_lgd)

def test_compute_realized_lgd_full_recovery():
    # Mock data: EAD = 100, full recovery, no costs
    ead = pd.Series([100])
    recoveries_pv = pd.Series([100])
    collection_costs_pv = pd.Series([0])

    # Expected LGD: (100 - 100 - 0) / 100 = 0
    expected_lgd = pd.Series([0.0])

    compute_realized_lgd.ead = ead
    compute_realized_lgd.recoveries_pv = recoveries_pv
    compute_realized_lgd.collection_costs_pv = collection_costs_pv

    lgd = compute_realized_lgd()
    pd.testing.assert_series_equal(lgd, expected_lgd)

def test_compute_realized_lgd_partial_recovery_and_costs():
    # Mock data: EAD = 100, partial recovery = 20, costs = 10
    ead = pd.Series([100])
    recoveries_pv = pd.Series([20])
    collection_costs_pv = pd.Series([10])

    # Expected LGD: (100 - 20 - 10) / 100 = 0.7
    expected_lgd = pd.Series([0.7])

    compute_realized_lgd.ead = ead
    compute_realized_lgd.recoveries_pv = recoveries_pv
    compute_realized_lgd.collection_costs_pv = collection_costs_pv

    lgd = compute_realized_lgd()
    pd.testing.assert_series_equal(lgd, expected_lgd)

def test_compute_realized_lgd_multiple_loans():
    # Mock data: Multiple loans with different EADs, recoveries, and costs
    ead = pd.Series([100, 200, 300])
    recoveries_pv = pd.Series([20, 50, 100])
    collection_costs_pv = pd.Series([10, 20, 30])

    # Expected LGDs:
    # Loan 1: (100 - 20 - 10) / 100 = 0.7
    # Loan 2: (200 - 50 - 20) / 200 = 0.65
    # Loan 3: (300 - 100 - 30) / 300 = 0.5666666666666667
    expected_lgd = pd.Series([0.7, 0.65, 0.5666666666666667])

    compute_realized_lgd.ead = ead
    compute_realized_lgd.recoveries_pv = recoveries_pv
    compute_realized_lgd.collection_costs_pv = collection_costs_pv

    lgd = compute_realized_lgd()
    pd.testing.assert_series_equal(lgd, expected_lgd)

def test_compute_realized_lgd_more_recoveries_than_ead():
    #Mock data
    ead = pd.Series([100])
    recoveries_pv = pd.Series([150])
    collection_costs_pv = pd.Series([0])
    expected_lgd = pd.Series([-0.5])

    compute_realized_lgd.ead = ead
    compute_realized_lgd.recoveries_pv = recoveries_pv
    compute_realized_lgd.collection_costs_pv = collection_costs_pv

    lgd = compute_realized_lgd()
    pd.testing.assert_series_equal(lgd, expected_lgd)
