import pytest
from definition_b68a4525c4404f94ae07adc555c947b1 import calculate_model_evaluation_metrics
import numpy as np

@pytest.mark.parametrize("y_true, y_pred, expected_keys", [
    ([0.1, 0.2, 0.3], [0.15, 0.25, 0.35], ['pseudo_r_squared', 'mae']),
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], ['pseudo_r_squared', 'mae']),
    ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], ['pseudo_r_squared', 'mae']),
    ([0.2, 0.4, 0.6], [0.3, 0.5, 0.7], ['pseudo_r_squared', 'mae']),
    ([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ['pseudo_r_squared', 'mae']),

])
def test_calculate_model_evaluation_metrics(y_true, y_pred, expected_keys):
    result = calculate_model_evaluation_metrics(y_true, y_pred)
    assert isinstance(result, dict)
    assert all(key in result for key in expected_keys)
    assert isinstance(result['pseudo_r_squared'], float)
    assert isinstance(result['mae'], float)

