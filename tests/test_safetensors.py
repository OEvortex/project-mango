"""
Tests for SafeTensors integration.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from mol import SAFETENSORS_AVAILABLE, MoLRuntime
from mol.core.mol_runtime import MoLConfig
from mol.training.trainer import TrainingConfig


class TestSafeTensorsManager:
    """Test SafeTensorsManager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        if not SAFETENSORS_AVAILABLE:
            pytest.skip("SafeTensors not available")
        
        self.config = MoLConfig(
            models=["gpt2", "distilgpt2"],
            adapter_type="linear",
            router_type="simple",
            max_layers=1,
            temperature=1.0
        )
    
    def create_simple_model(self):
        """Create a simple model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        return model
    
    def test_safetensors_manager_init(self):
        """Test SafeTensorsManager initialization."""
        from mol.utils.safetensors_utils import SafeTensorsManager
        
        manager = SafeTensorsManager()
        assert manager is not None
    
    def test_save_and_load_model_state(self):
        """Test saving and loading model state with SafeTensors."""
        from mol.utils.safetensors_utils import SafeTensorsManager
        
        manager = SafeTensorsManager()
        model = self.create_simple_model()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model"
            
            # Save model state
            metadata = {"test_key": "test_value", "version": "1.0"}
            manager.save_model_state(model, save_path, metadata)
            
            # Check file exists
            safetensors_file = save_path.with_suffix('.safetensors')
            assert safetensors_file.exists()
            
            # Load model state
            new_model = self.create_simple_model()
            loaded_metadata = manager.load_model_state(new_model, save_path)
            
            # Verify metadata
            assert loaded_metadata.get("test_key") == "test_value"
            assert loaded_metadata.get("version") == "1.0"
            
            # Verify model weights
            for orig_param, loaded_param in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(orig_param, loaded_param)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints with SafeTensors."""
        from mol.utils.safetensors_utils import SafeTensorsManager
        
        manager = SafeTensorsManager()
        model = self.create_simple_model()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 5,
            'loss': 0.123,
            'config': {'lr': 0.001}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "checkpoint"
            
            # Save checkpoint
            manager.save_checkpoint(checkpoint_data, save_path, use_safetensors=True)
            
            # Check files exist
            safetensors_file = save_path.with_suffix('.safetensors')
            aux_file = save_path.with_suffix('.aux.pt')
            assert safetensors_file.exists()
            assert aux_file.exists()
            
            # Load checkpoint
            loaded_data = manager.load_checkpoint(save_path)
            
            # Verify data
            assert loaded_data['epoch'] == 5
            assert abs(loaded_data['loss'] - 0.123) < 1e-6
            assert loaded_data['config']['lr'] == 0.001
            assert 'model_state_dict' in loaded_data
            assert 'optimizer_state_dict' in loaded_data
    
    def test_conversion_pytorch_to_safetensors(self):
        """Test converting PyTorch checkpoint to SafeTensors."""
        from mol.utils.safetensors_utils import SafeTensorsManager
        
        manager = SafeTensorsManager()
        model = self.create_simple_model()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pytorch_path = Path(temp_dir) / "model.pt"
            safetensors_path = Path(temp_dir) / "model.safetensors"
            
            # Save as PyTorch
            torch.save(model.state_dict(), pytorch_path)
            
            # Convert to SafeTensors
            metadata = {"converted": "true", "original_format": "pytorch"}
            manager.convert_pytorch_to_safetensors(
                pytorch_path, safetensors_path, metadata
            )
            
            # Check conversion
            assert safetensors_path.exists()
            
            # Load and verify
            new_model = self.create_simple_model()
            loaded_metadata = manager.load_model_state(new_model, safetensors_path.with_suffix(''))
            
            assert loaded_metadata.get("converted") == "true"
            
            # Verify weights match
            for orig_param, loaded_param in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(orig_param, loaded_param)
    
    def test_list_tensors(self):
        """Test listing tensors in SafeTensors file."""
        from mol.utils.safetensors_utils import SafeTensorsManager
        
        manager = SafeTensorsManager()
        model = self.create_simple_model()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model"
            
            # Save model
            manager.save_model_state(model, save_path)
            
            # List tensors
            safetensors_file = save_path.with_suffix('.safetensors')
            tensor_info = manager.list_tensors(safetensors_file)
            
            # Check tensor info
            assert len(tensor_info) > 0
            for name, info in tensor_info.items():
                assert 'shape' in info
                assert 'dtype' in info
                assert 'device' in info
    
    def test_get_file_info(self):
        """Test getting comprehensive file information."""
        from mol.utils.safetensors_utils import SafeTensorsManager
        
        manager = SafeTensorsManager()
        model = self.create_simple_model()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model"
            metadata = {"test": "data"}
            
            # Save model
            manager.save_model_state(model, save_path, metadata)
            
            # Get file info
            safetensors_file = save_path.with_suffix('.safetensors')
            file_info = manager.get_file_info(safetensors_file)
            
            # Check file info
            assert 'file_size' in file_info
            assert 'tensors' in file_info
            assert 'metadata' in file_info
            assert 'total_parameters' in file_info
            assert file_info['total_parameters'] > 0
    
    def test_fallback_to_pytorch(self):
        """Test fallback to PyTorch when SafeTensors fails."""
        from mol.utils.safetensors_utils import SafeTensorsManager
        
        manager = SafeTensorsManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model"
            
            # Mock SafeTensors to fail
            with patch('mol.utils.safetensors_utils.safe_save_file', side_effect=Exception("Mock error")):
                model = self.create_simple_model()
                
                # Should fallback to PyTorch
                manager.save_model_state(model, save_path)
                
                # Check PyTorch file exists
                pt_file = save_path.with_suffix('.pt')
                assert pt_file.exists()


class TestMoLRuntimeSafeTensors:
    """Test MoL runtime SafeTensors integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        if not SAFETENSORS_AVAILABLE:
            pytest.skip("SafeTensors not available")
        
        self.config = MoLConfig(
            models=["gpt2", "distilgpt2"],
            adapter_type="linear",
            router_type="simple",
            max_layers=1,
            temperature=1.0
        )
    
    def create_test_mol_runtime(self):
        """Create a test MoL runtime."""
        mol_runtime = MoLRuntime(self.config)
        
        # Mock expensive operations
        mol_runtime.tokenizer = Mock()
        mol_runtime.target_hidden_dim = 768
        mol_runtime.model_infos = {
            "gpt2": Mock(hidden_dim=768, num_layers=12),
            "distilgpt2": Mock(hidden_dim=768, num_layers=6)
        }
        
        return mol_runtime
    
    def test_mol_runtime_save_safetensors(self):
        """Test MoL runtime saving with SafeTensors."""
        mol_runtime = self.create_test_mol_runtime()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "mol_checkpoint"
            
            # Save with SafeTensors
            mol_runtime.save_checkpoint(str(save_path), use_safetensors=True)
            
            # Check file exists
            safetensors_file = save_path.with_suffix('.safetensors')
            assert safetensors_file.exists()
    
    def test_mol_runtime_save_pytorch(self):
        """Test MoL runtime saving with PyTorch."""
        mol_runtime = self.create_test_mol_runtime()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "mol_checkpoint.pt"
            
            # Save with PyTorch
            mol_runtime.save_checkpoint(str(save_path), use_safetensors=False)
            
            # Check file exists
            assert save_path.exists()
    
    def test_mol_runtime_load_safetensors(self):
        """Test MoL runtime loading with SafeTensors."""
        mol_runtime = self.create_test_mol_runtime()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "mol_checkpoint"
            
            # Save first
            mol_runtime.save_checkpoint(str(save_path), use_safetensors=True)
            
            # Load
            loaded_runtime = MoLRuntime.load_checkpoint(str(save_path))
            
            # Verify
            assert loaded_runtime.config.models == mol_runtime.config.models
            assert loaded_runtime.target_hidden_dim == mol_runtime.target_hidden_dim
    
    def test_mol_runtime_load_pytorch_fallback(self):
        """Test MoL runtime loading with PyTorch fallback."""
        mol_runtime = self.create_test_mol_runtime()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "mol_checkpoint.pt"
            
            # Save as PyTorch
            mol_runtime.save_checkpoint(str(save_path), use_safetensors=False)
            
            # Load (should work with both .pt extension and without)
            loaded_runtime = MoLRuntime.load_checkpoint(str(save_path))
            
            # Verify
            assert loaded_runtime.config.models == mol_runtime.config.models


class TestTrainerSafeTensors:
    """Test trainer SafeTensors integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        if not SAFETENSORS_AVAILABLE:
            pytest.skip("SafeTensors not available")
    
    def test_training_config_safetensors_default(self):
        """Test that SafeTensors is enabled by default in training config."""
        config = TrainingConfig()
        assert config.use_safetensors is True
    
    def test_training_config_safetensors_disabled(self):
        """Test disabling SafeTensors in training config."""
        config = TrainingConfig(use_safetensors=False)
        assert config.use_safetensors is False
    
    @patch('mol.training.trainer.MoLRuntime')
    def test_trainer_save_checkpoint_safetensors(self, mock_runtime_class):
        """Test trainer checkpoint saving with SafeTensors."""
        from mol.training.trainer import MoLTrainer, TrainingConfig
        
        # Mock MoL runtime
        mock_runtime = Mock()
        mock_runtime.state_dict.return_value = {"test": torch.tensor([1.0])}
        mock_runtime.parameters.return_value = [torch.tensor([1.0])]
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.state_dict.return_value = {"state": {}}
        
        config = TrainingConfig(use_safetensors=True, output_dir="./test_output")
        trainer = MoLTrainer(mock_runtime, config)
        trainer.optimizer = mock_optimizer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.output_dir = Path(temp_dir)
            
            # Mock safetensors_manager
            with patch('mol.training.trainer.safetensors_manager') as mock_manager:
                trainer.save_checkpoint()
                
                # Verify SafeTensors was used
                mock_manager.save_checkpoint.assert_called_once()
                args, kwargs = mock_manager.save_checkpoint.call_args
                assert kwargs.get('use_safetensors') is True
    
    @patch('mol.training.trainer.MoLRuntime')
    def test_trainer_load_checkpoint_safetensors(self, mock_runtime_class):
        """Test trainer checkpoint loading with SafeTensors."""
        from mol.training.trainer import MoLTrainer, TrainingConfig
        
        # Mock MoL runtime
        mock_runtime = Mock()
        mock_runtime.load_state_dict = Mock()
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.load_state_dict = Mock()
        
        config = TrainingConfig(use_safetensors=True)
        trainer = MoLTrainer(mock_runtime, config)
        trainer.optimizer = mock_optimizer
        
        # Mock checkpoint data
        mock_checkpoint = {
            'global_step': 100,
            'epoch': 5,
            'model_state_dict': {'test': torch.tensor([1.0])},
            'optimizer_state_dict': {'state': {}},
            'best_loss': 0.123,
            'config': config
        }
        
        with patch('mol.training.trainer.safetensors_manager') as mock_manager:
            mock_manager.load_checkpoint.return_value = mock_checkpoint
            
            trainer.load_checkpoint("test_checkpoint")
            
            # Verify SafeTensors was used
            mock_manager.load_checkpoint.assert_called_once_with("test_checkpoint", device='cpu')
            
            # Verify state was loaded
            mock_runtime.load_state_dict.assert_called_once()
            mock_optimizer.load_state_dict.assert_called_once()
            assert trainer.global_step == 100


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        if not SAFETENSORS_AVAILABLE:
            pytest.skip("SafeTensors not available")
    
    def test_is_safetensors_available(self):
        """Test SafeTensors availability check."""
        from mol.utils.safetensors_utils import is_safetensors_available
        
        available = is_safetensors_available()
        assert isinstance(available, bool)
        assert available is True  # Should be True since we skip if not available
    
    def test_save_load_model_safe(self):
        """Test convenience save/load functions."""
        from mol.utils.safetensors_utils import save_model_safe, load_model_safe
        
        model = nn.Linear(10, 5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model"
            metadata = {"test": "data"}
            
            # Save
            save_model_safe(model, save_path, metadata)
            
            # Check file exists
            safetensors_file = save_path.with_suffix('.safetensors')
            assert safetensors_file.exists()
            
            # Load
            new_model = nn.Linear(10, 5)
            loaded_metadata = load_model_safe(new_model, save_path)
            
            # Verify
            assert loaded_metadata.get("test") == "data"
            
            # Verify weights
            for orig_param, loaded_param in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(orig_param, loaded_param)


class TestErrorHandling:
    """Test error handling in SafeTensors integration."""
    
    def test_safetensors_not_available_fallback(self):
        """Test fallback when SafeTensors is not available."""
        with patch('mol.utils.safetensors_utils.SAFETENSORS_AVAILABLE', False):
            from mol.utils.safetensors_utils import SafeTensorsManager
            
            manager = SafeTensorsManager()
            model = nn.Linear(5, 2)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                save_path = Path(temp_dir) / "model"
                
                # Should fallback to PyTorch
                manager.save_model_state(model, save_path)
                
                # Check PyTorch file exists
                pt_file = save_path.with_suffix('.pt')
                assert pt_file.exists()
    
    def test_file_not_found_error(self):
        """Test file not found error handling."""
        from mol.utils.safetensors_utils import SafeTensorsManager
        
        manager = SafeTensorsManager()
        model = nn.Linear(5, 2)
        
        with pytest.raises(FileNotFoundError):
            manager.load_model_state(model, "nonexistent_file")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])