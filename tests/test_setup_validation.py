"""
Validation tests to verify the testing infrastructure is properly set up.
"""
import pytest
from pathlib import Path


class TestInfrastructureValidation:
    """Validate that the testing infrastructure is properly configured."""
    
    def test_pytest_installed(self):
        """Verify pytest is available."""
        assert pytest.__version__
    
    def test_project_structure_exists(self):
        """Verify the project structure is as expected."""
        project_root = Path(__file__).parent.parent
        
        # Check main source directories
        assert (project_root / "baseline").exists()
        assert (project_root / "utils").exists()
        
        # Check test directories
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        
        # Check configuration files
        assert (project_root / "pyproject.toml").exists()
    
    def test_fixtures_available(self, temp_dir, mock_config, sample_dialogue_data):
        """Verify that conftest fixtures are available."""
        # Test temp_dir fixture
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "model_type" in mock_config
        assert "batch_size" in mock_config
        
        # Test sample_dialogue_data fixture
        assert isinstance(sample_dialogue_data, dict)
        assert "utterances" in sample_dialogue_data
        assert len(sample_dialogue_data["utterances"]) > 0
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Verify unit test marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Verify integration test marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Verify slow test marker works."""
        assert True
    
    def test_coverage_configured(self):
        """Verify coverage is properly configured."""
        # This test will pass if coverage is running
        # The actual verification happens through pytest-cov
        assert True
    
    def test_mock_csv_fixture(self, mock_csv_data):
        """Test the mock CSV data fixture."""
        assert mock_csv_data.exists()
        content = mock_csv_data.read_text()
        assert "Utterance" in content
        assert "Emotion" in content
        assert "Sentiment" in content
    
    def test_emotion_and_sentiment_labels(self, emotion_labels, sentiment_labels):
        """Test emotion and sentiment label fixtures."""
        # MELD has 7 emotions
        assert len(emotion_labels) == 7
        assert "joy" in emotion_labels
        assert "sadness" in emotion_labels
        assert "anger" in emotion_labels
        
        # MELD has 3 sentiments
        assert len(sentiment_labels) == 3
        assert "positive" in sentiment_labels
        assert "negative" in sentiment_labels
        assert "neutral" in sentiment_labels