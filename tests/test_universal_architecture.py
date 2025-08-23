"""
Tests for Universal Architecture Handler.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from mol.core.universal_architecture import UniversalArchitectureHandler, ArchitectureInfo


class TestUniversalArchitectureHandler:
    """Test UniversalArchitectureHandler functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.handler = UniversalArchitectureHandler(trust_remote_code=False)
    
    def test_handler_initialization_secure_by_default(self):
        """Test that handler is secure by default."""
        handler = UniversalArchitectureHandler()
        assert handler.trust_remote_code is False
    
    def test_handler_initialization_with_trust_remote_code(self):
        """Test handler initialization with trust_remote_code=True."""
        with patch('mol.core.universal_architecture.logger') as mock_logger:
            handler = UniversalArchitectureHandler(trust_remote_code=True)
            assert handler.trust_remote_code is True
            mock_logger.warning.assert_called_once()
    
    def create_mock_config(self, model_type="gpt2", architectures=None, **kwargs):
        """Create a mock config object."""
        mock_config = Mock()
        mock_config.model_type = model_type
        mock_config.architectures = architectures or [f"{model_type.title()}ForCausalLM"]
        
        # Standard config attributes
        mock_config.hidden_size = kwargs.get('hidden_size', 768)
        mock_config.d_model = kwargs.get('d_model', None)
        mock_config.num_hidden_layers = kwargs.get('num_hidden_layers', 12)
        mock_config.num_layers = kwargs.get('num_layers', None)
        mock_config.num_attention_heads = kwargs.get('num_attention_heads', 12)
        mock_config.num_heads = kwargs.get('num_heads', None)
        mock_config.intermediate_size = kwargs.get('intermediate_size', 3072)
        mock_config.vocab_size = kwargs.get('vocab_size', 50257)
        mock_config.max_position_embeddings = kwargs.get('max_position_embeddings', 2048)
        mock_config.max_seq_length = kwargs.get('max_seq_length', None)
        mock_config.layer_norm_eps = kwargs.get('layer_norm_eps', 1e-5)
        mock_config.layernorm_epsilon = kwargs.get('layernorm_epsilon', None)
        
        return mock_config
    
    def test_detect_architecture_type_from_model_type(self):
        """Test architecture type detection from model_type."""
        config = self.create_mock_config(model_type="llama")
        arch_type = self.handler._detect_architecture_type(config)
        assert arch_type == "llama"
    
    def test_detect_architecture_type_from_architectures(self):
        """Test architecture type detection from architectures list."""
        config = self.create_mock_config(model_type=None, architectures=["LlamaForCausalLM"])
        config.model_type = None
        arch_type = self.handler._detect_architecture_type(config)
        assert arch_type == "llama"
    
    def test_classify_architecture_family_decoder_only(self):
        """Test classification of decoder-only architectures."""
        test_cases = [
            ("gpt2", "DECODER_ONLY"),
            ("llama", "DECODER_ONLY"),
            ("mistral", "DECODER_ONLY"),
            ("falcon", "DECODER_ONLY"),
            ("qwen", "DECODER_ONLY"),
        ]
        
        for arch_type, expected_family in test_cases:
            config = self.create_mock_config()
            family = self.handler._classify_architecture_family(config, arch_type)
            assert family == expected_family, f"Failed for {arch_type}"
    
    def test_classify_architecture_family_encoder_only(self):
        """Test classification of encoder-only architectures."""
        test_cases = [
            ("bert", "ENCODER_ONLY"),
            ("roberta", "ENCODER_ONLY"),
            ("electra", "ENCODER_ONLY"),
            ("deberta", "ENCODER_ONLY"),
            ("distilbert", "ENCODER_ONLY"),
        ]
        
        for arch_type, expected_family in test_cases:
            config = self.create_mock_config()
            family = self.handler._classify_architecture_family(config, arch_type)
            assert family == expected_family, f"Failed for {arch_type}"
    
    def test_classify_architecture_family_encoder_decoder(self):
        """Test classification of encoder-decoder architectures."""
        test_cases = [
            ("t5", "ENCODER_DECODER"),
            ("bart", "ENCODER_DECODER"),
            ("pegasus", "ENCODER_DECODER"),
            ("marian", "ENCODER_DECODER"),
        ]
        
        for arch_type, expected_family in test_cases:
            config = self.create_mock_config()
            family = self.handler._classify_architecture_family(config, arch_type)
            assert family == expected_family, f"Failed for {arch_type}"
    
    def test_classify_architecture_family_vision(self):
        """Test classification of vision architectures."""
        test_cases = [
            ("vit", "VISION"),
            ("deit", "VISION"),
            ("swin", "VISION"),
            ("beit", "VISION"),
        ]
        
        for arch_type, expected_family in test_cases:
            config = self.create_mock_config()
            family = self.handler._classify_architecture_family(config, arch_type)
            assert family == expected_family, f"Failed for {arch_type}"
    
    def test_classify_architecture_family_multimodal(self):
        """Test classification of multimodal architectures."""
        test_cases = [
            ("clip", "MULTIMODAL"),
            ("flava", "MULTIMODAL"),
            ("layoutlm", "MULTIMODAL"),
        ]
        
        for arch_type, expected_family in test_cases:
            config = self.create_mock_config()
            family = self.handler._classify_architecture_family(config, arch_type)
            assert family == expected_family, f"Failed for {arch_type}"
    
    def test_supports_causal_lm(self):
        """Test causal LM support detection."""
        assert self.handler._supports_causal_lm(None, "DECODER_ONLY") is True
        assert self.handler._supports_causal_lm(None, "ENCODER_DECODER") is True
        assert self.handler._supports_causal_lm(None, "ENCODER_ONLY") is False
        assert self.handler._supports_causal_lm(None, "VISION") is False
    
    def test_supports_masked_lm(self):
        """Test masked LM support detection."""
        assert self.handler._supports_masked_lm(None, "ENCODER_ONLY") is True
        assert self.handler._supports_masked_lm(None, "ENCODER_DECODER") is True
        assert self.handler._supports_masked_lm(None, "DECODER_ONLY") is False
        assert self.handler._supports_masked_lm(None, "VISION") is False
    
    def test_navigate_to_attribute_success(self):
        """Test successful attribute navigation."""
        mock_obj = Mock()
        mock_obj.transformer.h = "layers"
        
        result = self.handler._navigate_to_attribute(mock_obj, "transformer.h")
        assert result == "layers"
    
    def test_navigate_to_attribute_failure(self):
        """Test attribute navigation failure."""
        mock_obj = Mock()
        mock_obj.transformer = Mock()
        # Don't set 'h' attribute
        
        with pytest.raises(AttributeError):
            self.handler._navigate_to_attribute(mock_obj, "transformer.h")
    
    def test_validate_transformer_layers_success(self):
        """Test successful transformer layers validation."""
        # Create mock layers
        mock_layer = Mock()
        mock_layer.named_modules.return_value = [("attention", Mock()), ("mlp", Mock())]
        mock_layers = [mock_layer] * 12
        
        config = self.create_mock_config(num_hidden_layers=12)
        
        result = self.handler._validate_transformer_layers(mock_layers, config)
        assert result is True
    
    def test_validate_transformer_layers_wrong_length(self):
        """Test transformer layers validation with wrong length."""
        mock_layers = [Mock()] * 6  # Wrong number of layers
        config = self.create_mock_config(num_hidden_layers=12)
        
        result = self.handler._validate_transformer_layers(mock_layers, config)
        assert result is False
    
    def test_validate_embedding_layer_embedding_object(self):
        """Test embedding layer validation with nn.Embedding."""
        import torch.nn as nn
        embedding = nn.Embedding(1000, 768)
        config = self.create_mock_config()
        
        result = self.handler._validate_embedding_layer(embedding, config)
        assert result is True
    
    def test_validate_embedding_layer_with_weight(self):
        """Test embedding layer validation with weight tensor."""
        mock_embedding = Mock()
        mock_embedding.weight = Mock()
        mock_embedding.weight.shape = (50257, 768)  # vocab_size, hidden_size
        
        config = self.create_mock_config(vocab_size=50257)
        
        result = self.handler._validate_embedding_layer(mock_embedding, config)
        assert result is True
    
    def test_get_fallback_layer_path(self):
        """Test fallback layer path detection."""
        test_cases = [
            ("gpt2", "transformer.h"),
            ("llama", "model.layers"),
            ("bert", "encoder.layer"),
            ("t5", "encoder.block"),
            ("unknown", "transformer.h"),  # Default fallback
        ]
        
        for arch_type, expected_path in test_cases:
            path = self.handler._get_fallback_layer_path(arch_type)
            assert path == expected_path, f"Failed for {arch_type}"
    
    def test_get_fallback_embedding_path(self):
        """Test fallback embedding path detection."""
        test_cases = [
            ("bert", "embeddings"),
            ("gpt2", "transformer.wte"),
            ("llama", "model.embed_tokens"),
            ("t5", "shared"),
            ("unknown", "embeddings"),  # Default fallback
        ]
        
        for arch_type, expected_path in test_cases:
            config = self.create_mock_config(model_type=arch_type)
            path = self.handler._get_fallback_embedding_path(config)
            assert path == expected_path, f"Failed for {arch_type}"
    
    @patch('mol.core.universal_architecture.AutoConfig.from_pretrained')
    @patch('mol.core.universal_architecture.AutoModel.from_pretrained')
    def test_requires_remote_code_false(self, mock_auto_model, mock_auto_config):
        """Test remote code requirement detection - not required."""
        mock_auto_config.return_value = self.create_mock_config()
        mock_auto_model.return_value = Mock()
        
        config = self.create_mock_config()
        result = self.handler._requires_remote_code("test-model", config)
        assert result is False
    
    @patch('mol.core.universal_architecture.AutoConfig.from_pretrained')
    def test_requires_remote_code_true(self, mock_auto_config):
        """Test remote code requirement detection - required."""
        mock_auto_config.side_effect = Exception("Remote code required")
        
        config = self.create_mock_config()
        result = self.handler._requires_remote_code("test-model", config)
        assert result is True
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add some dummy entries
        self.handler.architecture_cache["test"] = Mock()
        
        self.handler.clear_cache()
        assert len(self.handler.architecture_cache) == 0
    
    @patch('mol.core.universal_architecture.AutoConfig.from_pretrained')
    def test_detect_architecture_caching(self, mock_auto_config):
        """Test architecture detection caching."""
        mock_config = self.create_mock_config()
        mock_auto_config.return_value = mock_config
        
        # Mock model loading to return None (no introspection)
        with patch.object(self.handler, '_load_model_for_introspection', return_value=None):
            # First call
            arch_info1 = self.handler.detect_architecture("test-model")
            
            # Second call should use cache
            arch_info2 = self.handler.detect_architecture("test-model")
            
            assert arch_info1 is arch_info2  # Same object from cache
            assert mock_auto_config.call_count == 1  # Only called once
    
    def test_architecture_info_creation(self):
        """Test ArchitectureInfo creation with proper attributes."""
        arch_info = ArchitectureInfo(
            model_name="test-model",
            architecture_family="DECODER_ONLY",
            architecture_type="gpt2",
            layer_path="transformer.h",
            embedding_path="transformer.wte",
            lm_head_path="lm_head",
            hidden_dim=768,
            num_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            vocab_size=50257,
            max_position_embeddings=2048,
            layer_norm_eps=1e-5,
            supports_causal_lm=True,
            supports_masked_lm=False,
            requires_remote_code=False
        )
        
        assert arch_info.model_name == "test-model"
        assert arch_info.architecture_family == "DECODER_ONLY"
        assert arch_info.hidden_dim == 768
        assert arch_info.supports_causal_lm is True
        assert arch_info.requires_remote_code is False


class TestUniversalArchitectureIntegration:
    """Integration tests for universal architecture handler."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.handler = UniversalArchitectureHandler(trust_remote_code=False)
    
    @patch('mol.core.universal_architecture.AutoConfig.from_pretrained')
    @patch('mol.core.universal_architecture.AutoModel.from_pretrained')
    def test_full_detection_workflow_gpt2(self, mock_auto_model, mock_auto_config):
        """Test full architecture detection workflow for GPT-2."""
        # Setup mocks
        mock_config = Mock()
        mock_config.model_type = "gpt2"
        mock_config.architectures = ["GPT2LMHeadModel"]
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12
        mock_config.num_attention_heads = 12
        mock_config.intermediate_size = 3072
        mock_config.vocab_size = 50257
        mock_config.max_position_embeddings = 1024
        mock_config.layer_norm_eps = 1e-5
        
        mock_auto_config.return_value = mock_config
        
        # Mock model with proper structure
        mock_model = Mock()
        mock_transformer = Mock()
        mock_layers = [Mock() for _ in range(12)]
        
        # Setup layer validation
        for layer in mock_layers:
            layer.named_modules.return_value = [("attention", Mock()), ("mlp", Mock())]
        
        mock_transformer.h = mock_layers
        mock_transformer.wte = Mock()  # Embeddings
        mock_model.transformer = mock_transformer
        mock_model.lm_head = Mock()
        
        mock_auto_model.return_value = mock_model
        
        # Test detection
        arch_info = self.handler.detect_architecture("gpt2")
        
        assert arch_info.model_name == "gpt2"
        assert arch_info.architecture_type == "gpt2"
        assert arch_info.architecture_family == "DECODER_ONLY"
        assert arch_info.layer_path == "transformer.h"
        assert arch_info.embedding_path == "transformer.wte"
        assert arch_info.hidden_dim == 768
        assert arch_info.num_layers == 12
        assert arch_info.supports_causal_lm is True
        assert arch_info.supports_masked_lm is False
    
    @patch('mol.core.universal_architecture.AutoConfig.from_pretrained')
    @patch('mol.core.universal_architecture.AutoModel.from_pretrained')
    def test_full_detection_workflow_bert(self, mock_auto_model, mock_auto_config):
        """Test full architecture detection workflow for BERT."""
        # Setup mocks
        mock_config = Mock()
        mock_config.model_type = "bert"
        mock_config.architectures = ["BertForMaskedLM"]
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12
        mock_config.num_attention_heads = 12
        mock_config.intermediate_size = 3072
        mock_config.vocab_size = 30522
        mock_config.max_position_embeddings = 512
        mock_config.layer_norm_eps = 1e-12
        
        mock_auto_config.return_value = mock_config
        
        # Mock model with proper structure
        mock_model = Mock()
        mock_encoder = Mock()
        mock_layers = [Mock() for _ in range(12)]
        
        # Setup layer validation
        for layer in mock_layers:
            layer.named_modules.return_value = [("attention", Mock()), ("intermediate", Mock())]
        
        mock_encoder.layer = mock_layers
        mock_model.encoder = mock_encoder
        mock_model.embeddings = Mock()  # Embeddings
        mock_model.cls = Mock()  # LM head for BERT
        
        mock_auto_model.return_value = mock_model
        
        # Test detection
        arch_info = self.handler.detect_architecture("bert-base-uncased")
        
        assert arch_info.model_name == "bert-base-uncased"
        assert arch_info.architecture_type == "bert"
        assert arch_info.architecture_family == "ENCODER_ONLY"
        assert arch_info.layer_path == "encoder.layer"
        assert arch_info.embedding_path == "embeddings"
        assert arch_info.hidden_dim == 768
        assert arch_info.num_layers == 12
        assert arch_info.supports_causal_lm is False
        assert arch_info.supports_masked_lm is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])