import pytest
import json
from definition_12bd8c4952a748698a021d86b1932320 import write_macro_scenarios_json

def test_write_macro_scenarios_json_empty_scenarios(tmp_path):
    filename = tmp_path / "test.json"
    scenarios = {}
    write_macro_scenarios_json(scenarios, filename)
    with open(filename, "r") as f:
        data = json.load(f)
    assert data == {}

def test_write_macro_scenarios_json_valid_scenarios(tmp_path):
    filename = tmp_path / "test.json"
    scenarios = {
        "scenario1": {"GDP": [1.0, 1.5, 2.0], "Unemployment": [5.0, 4.5, 4.0]},
        "scenario2": {"GDP": [-1.0, -0.5, 0.0], "Unemployment": [7.0, 7.5, 8.0]},
    }
    write_macro_scenarios_json(scenarios, filename)
    with open(filename, "r") as f:
        data = json.load(f)
    assert data == scenarios

def test_write_macro_scenarios_json_invalid_filename(tmp_path):
    filename = 123  # Invalid filename type
    scenarios = {"scenario1": {"GDP": [1.0], "Unemployment": [5.0]}}
    with pytest.raises(TypeError):
        write_macro_scenarios_json(scenarios, filename)

def test_write_macro_scenarios_json_non_serializable_data(tmp_path):
    filename = tmp_path / "test.json"
    scenarios = {"scenario1": {"GDP": [1.0], "Unemployment": [complex(1, 1)]}} # complex is not JSON serializable
    with pytest.raises(TypeError):
        write_macro_scenarios_json(scenarios, filename)

def test_write_macro_scenarios_json_file_creation(tmp_path):
    filename = tmp_path / "new_file.json"
    scenarios = {"scenario1": {"GDP": [1.0], "Unemployment": [5.0]}}
    write_macro_scenarios_json(scenarios, filename)
    assert filename.exists()
    with open(filename, "r") as f:
        data = json.load(f)
    assert data == scenarios
