import pytest
import pandas as pd
from definition_44af6493b0834fc49860ef59fc5c6491 import compute_ead

def create_loan_row(total_pymnt, funded_amnt):
    return pd.Series({'total_pymnt': total_pymnt, 'funded_amnt': funded_amnt})

def create_loan_row_with_nan(total_pymnt, funded_amnt):
    return pd.Series({'total_pymnt': total_pymnt, 'funded_amnt': funded_amnt})

def create_loan_row_with_negative(total_pymnt, funded_amnt):
    return pd.Series({'total_pymnt': total_pymnt, 'funded_amnt': funded_amnt})

def create_loan_row_zero_funded(total_pymnt, funded_amnt):
    return pd.Series({'total_pymnt': total_pymnt, 'funded_amnt': funded_amnt})

def create_loan_row_string_values(total_pymnt, funded_amnt):
    return pd.Series({'total_pymnt': total_pymnt, 'funded_amnt': funded_amnt})


def test_compute_ead_normal_case():
    row = create_loan_row(total_pymnt=5000, funded_amnt=10000)
    assert compute_ead(row) == 10000.0

def test_compute_ead_nan_values():
    row = create_loan_row_with_nan(total_pymnt=float('nan'), funded_amnt=10000)
    assert compute_ead(row) == 10000.0

def test_compute_ead_negative_funded_amnt():
    row = create_loan_row_with_negative(total_pymnt=5000, funded_amnt=-10000)
    assert compute_ead(row) == -10000.0

def test_compute_ead_zero_funded_amnt():
    row = create_loan_row_zero_funded(total_pymnt=5000, funded_amnt=0)
    assert compute_ead(row) == 0.0

def test_compute_ead_string_values():
    row = create_loan_row_string_values(total_pymnt="5000", funded_amnt="10000")
    with pytest.raises(TypeError):
        compute_ead(row)
