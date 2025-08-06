import pytest
import pandas as pd
import numpy as np
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from definition_ae0aba19bfbc46e9bc6bc075aa1cda64 import fit_beta_regression


def test_fit_beta_regression_typical( ):
    X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [5, 4, 3, 2, 1]})
    y = pd.Series([0.2, 0.4, 0.6, 0.8, 0.9])
    model = fit_beta_regression(X, y)
    assert isinstance(model, GLM)
    assert isinstance(model.family, families.Beta)

def test_fit_beta_regression_zero_inflated():
    X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [5, 4, 3, 2, 1]})
    y = pd.Series([0.0, 0.4, 0.6, 0.8, 0.9])
    model = fit_beta_regression(X, y)
    assert isinstance(model, GLM)
    assert isinstance(model.family, families.Beta)

def test_fit_beta_regression_one_inflated():
    X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [5, 4, 3, 2, 1]})
    y = pd.Series([0.2, 0.4, 0.6, 0.8, 1.0])
    model = fit_beta_regression(X, y)
    assert isinstance(model, GLM)
    assert isinstance(model.family, families.Beta)

def test_fit_beta_regression_empty_data():
    X = pd.DataFrame({'feature1': [], 'feature2': []})
    y = pd.Series([])
    model = fit_beta_regression(X, y)
    assert isinstance(model, GLM)
    assert isinstance(model.family, families.Beta)

def test_fit_beta_regression_constant_y():
    X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [5, 4, 3, 2, 1]})
    y = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])
    model = fit_beta_regression(X, y)
    assert isinstance(model, GLM)
    assert isinstance(model.family, families.Beta)
