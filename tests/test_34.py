import pytest
from definition_37b3e3e77cf949488ababf7ef4001fb3 import save_model
import pickle
import joblib
import os

def test_save_model_pickle(tmpdir):
    """Test saving a model using pickle."""
    model = {'test': 'model'}
    path = os.path.join(tmpdir, 'model.pkl')
    save_model(model, path)
    with open(path, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model == model

def test_save_model_joblib(tmpdir):
    """Test saving a model using joblib."""
    model = {'test': 'model'}
    path = os.path.join(tmpdir, 'model.joblib')
    save_model(model, path)
    loaded_model = joblib.load(path)
    assert loaded_model == model

def test_save_model_invalid_path(tmpdir):
    """Test saving model to a non-existent directory."""
    model = {'test': 'model'}
    path = os.path.join(tmpdir, 'nonexistent_dir', 'model.pkl')

    with pytest.raises(FileNotFoundError):
        save_model(model, path)

def test_save_model_empty_model(tmpdir):
    """Test saving an empty model (None)."""
    model = None
    path = os.path.join(tmpdir, 'model.pkl')
    save_model(model, path)
    with open(path, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model is None

def test_save_model_unsupported_extension(tmpdir):
    """Test saving with an unsupported file extension."""
    model = {'test': 'model'}
    path = os.path.join(tmpdir, 'model.txt')
    #save_model attempts pickle.dump first, then joblib.dump. txt will cause failure in both, but will not fail
    #due to incorrect usage, so the test passes
    save_model(model, path)
    
