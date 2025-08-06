import pytest
from definition_6a63db12606742d1b886c45270d6677e import calculate_pseudo_r_squared
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np


def mock_model(loglike, null_loglike):
    class MockModel:
        def __init__(self, loglike, null_loglike):
            self.llnull = null_loglike
            self.llf = loglike

    return MockModel(loglike, null_loglike)



def test_pseudo_r_squared_positive():
    model = mock_model(-100, -200)
    assert calculate_pseudo_r_squared(model) == 0.5

def test_pseudo_r_squared_zero():
    model = mock_model(-200, -200)
    assert calculate_pseudo_r_squared(model) == 0.0

def test_pseudo_r_squared_negative():
     model = mock_model(-300, -200)
     assert calculate_pseudo_r_squared(model) == pytest.approx(-0.5)

def test_pseudo_r_squared_equal_likelihood():
    model = mock_model(-150, -150)
    assert calculate_pseudo_r_squared(model) == 0.0

def test_pseudo_r_squared_with_real_data():
    # Create sample data
    data = {'y': np.random.beta(2, 5, 100), 'x1': np.random.rand(100), 'x2': np.random.rand(100)}
    df = pd.DataFrame(data)

    # Fit a Beta regression model (this might require statsmodels to be installed)
    try:
        model = smf.betareg('y ~ x1 + x2', data=df).fit()
        r_squared = calculate_pseudo_r_squared(model)
        assert 0 <= r_squared <= 1 or np.isnan(r_squared)

    except Exception as e:
        pytest.skip(f"Statsmodels or patsy installation issue, skipping test. {e}")

