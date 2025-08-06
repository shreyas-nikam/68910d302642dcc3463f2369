import pytest
import json
from definition_7bbca5e007ef45d08bd48e3e9cf34351 import write_macro_scenarios_json

def test_write_macro_scenarios_json_empty_scenarios(tmp_path):
    """Test writing an empty dictionary to a JSON file."""
    scenarios = {}
    file_path = tmp_path / "empty_scenarios.json"
    write_macro_scenarios_json(scenarios, str(file_path))
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == {}

def test_write_macro_scenarios_json_valid_scenarios(tmp_path):
    """Test writing valid macroeconomic scenarios to a JSON file."""
    scenarios = {
        "baseline": {"gdp_growth": 2.5, "unemployment_rate": 4.0},
        "adverse": {"gdp_growth": -1.0, "unemployment_rate": 7.0},
    }
    file_path = tmp_path / "valid_scenarios.json"
    write_macro_scenarios_json(scenarios, str(file_path))
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == scenarios

def test_write_macro_scenarios_json_non_string_path(tmp_path):
    """Test that a TypeError is raised if the path is not a string."""
    scenarios = {"scenario1": {"var1": 1, "var2": 2}}
    file_path = tmp_path / "non_string_path.json"
    with pytest.raises(TypeError):
        write_macro_scenarios_json(scenarios, file_path)

def test_write_macro_scenarios_json_invalid_scenario_data(tmp_path):
    """Test writing scenarios with non-numeric data."""
    scenarios = {
        "baseline": {"gdp_growth": "high", "unemployment_rate": 4.0},
    }
    file_path = tmp_path / "invalid_scenarios.json"
    # In this case the scenarios are still saved and no exception is thrown.
    # Here we are testing for exceptions that prevent writing.
    write_macro_scenarios_json(scenarios, str(file_path))

    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == scenarios

def test_write_macro_scenarios_json_file_already_exists(tmp_path):
    """Test overwriting an existing JSON file."""
    file_path = tmp_path / "existing_file.json"
    with open(file_path, "w") as f:
        json.dump({"initial": "data"}, f)

    scenarios = {"new": "data"}
    write_macro_scenarios_json(scenarios, str(file_path))

    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == scenarios
