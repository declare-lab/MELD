"""
Shared pytest fixtures for MELD testing.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide mock configuration for testing."""
    return {
        "model_type": "baseline",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "hidden_size": 128,
        "dropout": 0.2,
        "num_classes": 7,  # 7 emotions in MELD
    }


@pytest.fixture
def sample_dialogue_data() -> Dict[str, Any]:
    """Provide sample dialogue data for testing."""
    return {
        "utterances": [
            {
                "text": "How are you feeling today?",
                "speaker": "A",
                "emotion": "neutral",
                "sentiment": "neutral"
            },
            {
                "text": "I'm feeling great, thanks!",
                "speaker": "B",
                "emotion": "joy",
                "sentiment": "positive"
            },
            {
                "text": "That's wonderful to hear!",
                "speaker": "A",
                "emotion": "joy",
                "sentiment": "positive"
            }
        ],
        "dialogue_id": "test_dialogue_001",
        "scene_id": "s01e01"
    }


@pytest.fixture
def emotion_labels() -> list:
    """Provide the list of emotion labels used in MELD."""
    return ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]


@pytest.fixture
def sentiment_labels() -> list:
    """Provide the list of sentiment labels used in MELD."""
    return ["positive", "negative", "neutral"]


@pytest.fixture
def mock_csv_data(temp_dir: Path) -> Path:
    """Create a mock CSV file for testing data loading."""
    csv_path = temp_dir / "test_data.csv"
    csv_content = """Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID
1,"Hey, how are you?",Monica,neutral,neutral,0,0
2,"I'm doing great!",Rachel,joy,positive,0,1
3,"That's wonderful!",Monica,joy,positive,0,2
4,"What's wrong?",Ross,neutral,neutral,1,0
5,"I'm feeling sad.",Chandler,sadness,negative,1,1
"""
    csv_path.write_text(csv_content)
    return csv_path


@pytest.fixture
def mock_pickle_data() -> Dict[str, Any]:
    """Provide mock data that would typically be loaded from pickle files."""
    return {
        "trainId": list(range(100)),
        "trainSpeaker": [["A", "B", "A"] for _ in range(100)],
        "trainUtterance": [["Hello", "Hi there", "How are you?"] for _ in range(100)],
        "trainEmotion": [[0, 3, 0] for _ in range(100)],  # neutral, joy, neutral
        "trainSentiment": [[2, 0, 2] for _ in range(100)],  # neutral, positive, neutral
    }


@pytest.fixture
def data_path(temp_dir: Path) -> Path:
    """Create a mock data directory structure."""
    data_dir = temp_dir / "data"
    meld_dir = data_dir / "MELD"
    meld_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_stdout(monkeypatch) -> list:
    """Capture stdout for testing print statements."""
    captured = []
    
    def mock_print(*args, **kwargs):
        captured.append(" ".join(str(arg) for arg in args))
    
    monkeypatch.setattr("builtins.print", mock_print)
    return captured


@pytest.fixture(scope="session")
def test_resources_dir() -> Path:
    """Path to test resources directory."""
    return Path(__file__).parent / "resources"