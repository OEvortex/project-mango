"""
Tests for adapter implementations.
"""

import torch
import pytest
from mol.core.adapters import LinearAdapter, BottleneckAdapter, create_adapter


class TestLinearAdapter:
    """Test cases for LinearAdapter."""
    
    def test_same_dimension_identity(self):
        """Test identity initialization for same dimensions."""
        adapter = LinearAdapter(
            input_dim=512,
            output_dim=512,
            init_identity=True,
            use_residual=True
        )
        
        # Create test input
        x = torch.randn(2, 10, 512)  # [batch, seq, hidden]
        
        # Forward pass should approximate identity initially
        output = adapter(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check that it's close to identity (with some tolerance for LayerNorm)
        diff = torch.abs(output - x).mean()
        assert diff < 0.5, f"Output too different from input: {diff}"
    
    def test_different_dimensions(self):
        """Test adapter with different input/output dimensions."""
        adapter = LinearAdapter(
            input_dim=512,
            output_dim=768,
            init_identity=True
        )
        
        x = torch.randn(2, 10, 512)
        output = adapter(x)
        
        # Check output shape
        assert output.shape == (2, 10, 768)
        
        # Check that output is not all zeros
        assert output.abs().sum() > 0
    
    def test_no_layer_norm(self):
        """Test adapter without layer normalization."""
        adapter = LinearAdapter(
            input_dim=512,
            output_dim=512,
            use_layer_norm=False,
            init_identity=True
        )
        
        x = torch.randn(2, 10, 512)
        output = adapter(x)
        
        # Should be very close to identity without LayerNorm
        diff = torch.abs(output - x).mean()
        assert diff < 0.01, f"Should be close to identity: {diff}"
    
    def test_no_residual(self):
        """Test adapter without residual connection."""
        adapter = LinearAdapter(
            input_dim=512,
            output_dim=768,
            use_residual=False,
            init_identity=False
        )
        
        x = torch.randn(2, 10, 512)
        output = adapter(x)
        
        assert output.shape == (2, 10, 768)
        assert not torch.allclose(output[:, :, :512], x, atol=1e-3)


class TestBottleneckAdapter:
    """Test cases for BottleneckAdapter."""
    
    def test_bottleneck_dimensions(self):
        """Test bottleneck adapter with custom bottleneck dimension."""
        adapter = BottleneckAdapter(
            input_dim=768,
            output_dim=768,
            bottleneck_dim=128,
            init_identity=True
        )
        
        x = torch.randn(2, 10, 768)
        output = adapter(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check bottleneck dimension
        assert adapter.bottleneck_dim == 128
    
    def test_identity_initialization(self):
        """Test that identity initialization produces near-zero output initially."""
        adapter = BottleneckAdapter(
            input_dim=512,
            output_dim=512,
            init_identity=True
        )
        
        x = torch.randn(2, 10, 512)
        output = adapter(x)
        
        # With identity initialization + residual, output should be close to input
        diff = torch.abs(output - x).mean()
        assert diff < 0.5, f"Identity initialization not working: {diff}"
    
    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ["gelu", "relu", "silu"]:
            adapter = BottleneckAdapter(
                input_dim=512,
                output_dim=512,
                activation=activation
            )
            
            x = torch.randn(2, 10, 512)
            output = adapter(x)
            
            assert output.shape == x.shape
            assert not torch.isnan(output).any()
    
    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError):
            BottleneckAdapter(
                input_dim=512,
                output_dim=512,
                activation="invalid_activation"
            )


class TestAdapterFactory:
    """Test adapter factory function."""
    
    def test_create_linear_adapter(self):
        """Test creating linear adapter via factory."""
        adapter = create_adapter("linear", 512, 768)
        assert isinstance(adapter, LinearAdapter)
        assert adapter.input_dim == 512
        assert adapter.output_dim == 768
    
    def test_create_bottleneck_adapter(self):
        """Test creating bottleneck adapter via factory."""
        adapter = create_adapter("bottleneck", 512, 768, bottleneck_dim=128)
        assert isinstance(adapter, BottleneckAdapter)
        assert adapter.input_dim == 512
        assert adapter.output_dim == 768
        assert adapter.bottleneck_dim == 128
    
    def test_invalid_adapter_type(self):
        """Test that invalid adapter type raises error."""
        with pytest.raises(ValueError):
            create_adapter("invalid_type", 512, 768)


def test_adapter_forward_backward():
    """Test that adapters work in training mode with gradients."""
    adapter = LinearAdapter(512, 768)
    adapter.train()
    
    x = torch.randn(2, 10, 512, requires_grad=True)
    output = adapter(x)
    
    # Compute dummy loss and backward pass
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert adapter.projection.weight.grad is not None


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running adapter tests...")
    
    test_adapter = TestLinearAdapter()
    test_adapter.test_same_dimension_identity()
    test_adapter.test_different_dimensions()
    print("✅ LinearAdapter tests passed")
    
    test_bottleneck = TestBottleneckAdapter()
    test_bottleneck.test_bottleneck_dimensions()
    test_bottleneck.test_identity_initialization()
    test_bottleneck.test_different_activations()
    print("✅ BottleneckAdapter tests passed")
    
    test_factory = TestAdapterFactory()
    test_factory.test_create_linear_adapter()
    test_factory.test_create_bottleneck_adapter()
    print("✅ Adapter factory tests passed")
    
    test_adapter_forward_backward()
    print("✅ Forward/backward tests passed")
    
    print("\n🎉 All adapter tests passed!")