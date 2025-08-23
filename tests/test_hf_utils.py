"""
Tests for HuggingFace Hub integration.
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from mol import MoLRuntime, HF_AVAILABLE
from mol.core.mol_runtime import MoLConfig


class TestHuggingFaceIntegration:
    """Test HuggingFace Hub integration functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = MoLConfig(
            models=["gpt2", "distilgpt2"],
            adapter_type="linear",
            router_type="simple",
            max_layers=2,
            temperature=1.0
        )
        
        # Mock HuggingFace components if not available
        if not HF_AVAILABLE:
            pytest.skip("HuggingFace Hub not available")
    
    def create_test_mol_runtime(self):
        """Create a minimal MoL runtime for testing."""
        mol_runtime = MoLRuntime(self.config)
        
        # Mock the expensive operations
        mol_runtime.tokenizer = Mock()
        mol_runtime.tokenizer.save_pretrained = Mock()
        
        # Add minimal layers for testing
        mol_runtime.target_hidden_dim = 768
        mol_runtime.model_infos = {
            "gpt2": Mock(hidden_dim=768, num_layers=12),
            "distilgpt2": Mock(hidden_dim=768, num_layers=6)
        }
        
        return mol_runtime
    
    @patch('mol.utils.hf_utils.HfApi')
    @patch('mol.utils.hf_utils.AutoConfig')
    def test_hf_publisher_init(self, mock_config, mock_api):
        """Test HuggingFacePublisher initialization."""
        from mol.utils.hf_utils import HuggingFacePublisher
        
        publisher = HuggingFacePublisher(token="test_token")
        assert publisher.token == "test_token"
        mock_api.assert_called_once_with(token="test_token")
    
    @patch('mol.utils.hf_utils.HfApi')
    @patch('mol.utils.hf_utils.tempfile.TemporaryDirectory')
    def test_push_mol_runtime(self, mock_tempdir, mock_api):
        """Test pushing MoL runtime to HuggingFace."""
        from mol.utils.hf_utils import HuggingFacePublisher
        
        # Setup mocks
        mock_temp_path = Mock()
        mock_temp_path.__truediv__ = lambda self, x: Mock()
        mock_tempdir.return_value.__enter__.return_value = "/tmp/test"
        
        mock_api_instance = Mock()
        mock_api.return_value = mock_api_instance
        
        # Create test runtime
        mol_runtime = self.create_test_mol_runtime()
        mol_runtime.save_checkpoint = Mock()
        
        # Test push
        publisher = HuggingFacePublisher()
        with patch('pathlib.Path', return_value=mock_temp_path):
            result = publisher.push_mol_runtime(
                mol_runtime,
                "test-user/test-model",
                commit_message="Test upload"
            )
        
        # Verify API calls
        mock_api_instance.create_repo.assert_called_once()
        mock_api_instance.upload_folder.assert_called_once()
        assert result == "https://huggingface.co/test-user/test-model"
    
    @patch('mol.utils.hf_utils.HfApi')
    @patch('mol.utils.hf_utils.tempfile.TemporaryDirectory')
    def test_push_fused_model(self, mock_tempdir, mock_api):
        """Test pushing fully fused model to HuggingFace."""
        from mol.utils.hf_utils import HuggingFacePublisher
        
        # Setup mocks
        mock_temp_path = Mock()
        mock_temp_path.__truediv__ = lambda self, x: Mock()
        mock_tempdir.return_value.__enter__.return_value = "/tmp/test"
        
        mock_api_instance = Mock()
        mock_api.return_value = mock_api_instance
        
        # Create test runtime
        mol_runtime = self.create_test_mol_runtime()
        
        # Mock fused model creation
        publisher = HuggingFacePublisher()
        with patch.object(publisher, '_create_fused_model') as mock_create:
            mock_model = Mock()
            mock_model.save_pretrained = Mock()
            mock_create.return_value = mock_model
            
            with patch('pathlib.Path', return_value=mock_temp_path):
                result = publisher.push_fused_model(
                    mol_runtime,
                    "test-user/test-fused",
                    fusion_method="weighted_average"
                )
        
        # Verify calls
        mock_create.assert_called_once_with(mol_runtime, "weighted_average")
        mock_api_instance.create_repo.assert_called_once()
        mock_api_instance.upload_folder.assert_called_once()
        assert result == "https://huggingface.co/test-user/test-fused"
    
    def test_fusion_methods(self):
        """Test different fusion methods."""
        from mol.utils.hf_utils import HuggingFacePublisher
        
        mol_runtime = self.create_test_mol_runtime()
        publisher = HuggingFacePublisher()
        
        # Mock block extractor
        with patch('mol.utils.hf_utils.BlockExtractor') as mock_extractor:
            mock_extractor_instance = Mock()
            mock_model = Mock()
            mock_model.config = Mock()
            mock_model.state_dict.return_value = {"test": torch.tensor([1.0])}
            mock_extractor_instance.load_model.return_value = (mock_model, Mock())
            mock_extractor.return_value = mock_extractor_instance
            
            # Test weighted average
            fused = publisher._create_fused_model(mol_runtime, "weighted_average")
            assert fused is not None
            
            # Test best expert
            fused = publisher._create_fused_model(mol_runtime, "best_expert")
            assert fused is not None
            
            # Test invalid method
            with pytest.raises(ValueError):
                publisher._create_fused_model(mol_runtime, "invalid_method")
    
    def test_mol_runtime_push_methods(self):
        """Test MoLRuntime push_to_hf methods."""
        mol_runtime = self.create_test_mol_runtime()
        
        # Mock HuggingFacePublisher
        with patch('mol.utils.hf_utils.HuggingFacePublisher') as mock_publisher_class:
            mock_publisher = Mock()
            mock_publisher.push_mol_runtime.return_value = "https://test.com"
            mock_publisher.push_fused_model.return_value = "https://test.com"
            mock_publisher_class.return_value = mock_publisher
            
            # Test runtime push
            result = mol_runtime.push_to_hf(
                "test/repo",
                fusion_type="runtime"
            )
            assert result == "https://test.com"
            mock_publisher.push_mol_runtime.assert_called_once()
            
            # Test fused push
            result = mol_runtime.push_to_hf(
                "test/repo",
                fusion_type="fused",
                fusion_method="weighted_average"
            )
            assert result == "https://test.com"
            mock_publisher.push_fused_model.assert_called_once()
            
            # Test invalid type
            with pytest.raises(ValueError):
                mol_runtime.push_to_hf("test/repo", fusion_type="invalid")
    
    def test_create_fused_model_method(self):
        """Test MoLRuntime create_fused_model method."""
        mol_runtime = self.create_test_mol_runtime()
        
        with patch('mol.utils.hf_utils.HuggingFacePublisher') as mock_publisher_class:
            mock_publisher = Mock()
            mock_model = Mock()
            mock_publisher._create_fused_model.return_value = mock_model
            mock_publisher_class.return_value = mock_publisher
            
            result = mol_runtime.create_fused_model("weighted_average")
            assert result == mock_model
            mock_publisher._create_fused_model.assert_called_once_with(
                mol_runtime, "weighted_average"
            )
    
    def test_model_card_creation(self):
        """Test model card generation."""
        from mol.utils.hf_utils import HuggingFacePublisher
        
        mol_runtime = self.create_test_mol_runtime()
        publisher = HuggingFacePublisher()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test runtime model card
            publisher._create_mol_model_card(
                temp_path, mol_runtime, is_runtime=True
            )
            
            readme_path = temp_path / "README.md"
            assert readme_path.exists()
            
            content = readme_path.read_text()
            assert "MoL Runtime Model" in content
            assert "dynamic-routing" in content
            
            # Test fused model card
            publisher._create_mol_model_card(
                temp_path, mol_runtime, is_runtime=False, 
                fusion_method="weighted_average"
            )
            
            content = readme_path.read_text()
            assert "MoL Fused Model" in content
            assert "weighted_average" in content
    
    def test_config_creation(self):
        """Test MoL config file creation."""
        from mol.utils.hf_utils import HuggingFacePublisher
        
        mol_runtime = self.create_test_mol_runtime()
        publisher = HuggingFacePublisher()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            publisher._create_mol_config(temp_path, mol_runtime)
            
            config_path = temp_path / "mol_config.json"
            assert config_path.exists()
            
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            assert config_data["mol_runtime"] is True
            assert config_data["models"] == ["gpt2", "distilgpt2"]
            assert config_data["adapter_type"] == "linear"
    
    def test_error_handling(self):
        """Test error handling in HF integration."""
        from mol.utils.hf_utils import HuggingFacePublisher
        
        mol_runtime = self.create_test_mol_runtime()
        
        # Test initialization without HF hub
        with patch('mol.utils.hf_utils.HF_HUB_AVAILABLE', False):
            with pytest.raises(ImportError):
                HuggingFacePublisher()
        
        # Test API errors
        with patch('mol.utils.hf_utils.HfApi') as mock_api:
            mock_api.return_value.create_repo.side_effect = Exception("API Error")
            mock_api.return_value.upload_folder.side_effect = Exception("Upload Error")
            
            publisher = HuggingFacePublisher()
            
            # Should handle repo creation errors gracefully
            with patch('mol.utils.hf_utils.tempfile.TemporaryDirectory'):
                with patch('pathlib.Path'):
                    with pytest.raises(Exception):
                        publisher.push_mol_runtime(mol_runtime, "test/repo")


class TestConvenienceFunction:
    """Test the convenience push_mol_to_hf function."""
    
    def test_push_mol_to_hf_runtime(self):
        """Test convenience function for runtime push."""
        from mol.utils.hf_utils import push_mol_to_hf
        
        mol_runtime = Mock()
        mol_runtime.push_to_hf.return_value = "https://test.com"
        
        result = push_mol_to_hf(
            mol_runtime,
            "test/repo",
            fusion_type="runtime",
            token="test_token"
        )
        
        assert result == "https://test.com"
        mol_runtime.push_to_hf.assert_called_once_with(
            "test/repo",
            fusion_type="runtime",
            token="test_token"
        )
    
    def test_push_mol_to_hf_fused(self):
        """Test convenience function for fused push."""
        from mol.utils.hf_utils import push_mol_to_hf
        
        mol_runtime = Mock()
        mol_runtime.push_to_hf.return_value = "https://test.com"
        
        result = push_mol_to_hf(
            mol_runtime,
            "test/repo",
            fusion_type="fused",
            fusion_method="best_expert"
        )
        
        assert result == "https://test.com"
        mol_runtime.push_to_hf.assert_called_once_with(
            "test/repo",
            fusion_type="fused",
            fusion_method="best_expert"
        )
    
    def test_push_mol_to_hf_invalid_type(self):
        """Test convenience function with invalid fusion type."""
        from mol.utils.hf_utils import push_mol_to_hf
        
        mol_runtime = Mock()
        
        with pytest.raises(ValueError):
            push_mol_to_hf(mol_runtime, "test/repo", fusion_type="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])