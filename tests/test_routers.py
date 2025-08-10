"""
Tests for router implementations.
"""

import torch
import pytest
from mol.core.routers import (
    SimpleRouter, TokenLevelRouter, create_router,
    compute_router_entropy, compute_load_balancing_loss
)


class TestSimpleRouter:
    """Test cases for SimpleRouter."""
    
    def test_pooled_routing_mean(self):
        """Test mean pooling router."""
        router = SimpleRouter(
            hidden_dim=512,
            num_experts=3,
            pooling_type="mean",
            temperature=1.0
        )
        
        x = torch.randn(2, 10, 512)  # [batch, seq, hidden]
        expert_weights, router_logits = router(x)
        
        # Check shapes
        assert expert_weights.shape == (2, 10, 3)  # [batch, seq, num_experts]
        assert router_logits.shape == (2, 3)  # [batch, num_experts]
        
        # Check that weights sum to 1 along expert dimension
        weights_sum = expert_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)
        
        # All tokens in sequence should have same routing (pooled)
        for b in range(2):
            for e in range(3):
                assert torch.allclose(
                    expert_weights[b, :, e], 
                    expert_weights[b, 0, e] * torch.ones(10)
                )
    
    def test_attention_pooling(self):
        """Test attention-based pooling."""
        router = SimpleRouter(
            hidden_dim=512,
            num_experts=2,
            pooling_type="attention"
        )
        
        x = torch.randn(2, 10, 512)
        expert_weights, router_logits = router(x)
        
        assert expert_weights.shape == (2, 10, 2)
        assert router_logits.shape == (2, 2)
        
        # Check that weights are valid probabilities
        assert (expert_weights >= 0).all()
        assert torch.allclose(expert_weights.sum(dim=-1), torch.ones(2, 10), atol=1e-5)
    
    def test_with_attention_mask(self):
        """Test router with attention mask."""
        router = SimpleRouter(hidden_dim=512, num_experts=2)
        
        x = torch.randn(2, 10, 512)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
        
        expert_weights, router_logits = router(x, attention_mask)
        
        assert expert_weights.shape == (2, 10, 2)
        
        # Weights should still sum to 1 (pooled routing ignores mask for weight computation)
        weights_sum = expert_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)


class TestTokenLevelRouter:
    """Test cases for TokenLevelRouter."""
    
    def test_token_level_routing(self):
        """Test per-token routing decisions."""
        router = TokenLevelRouter(
            hidden_dim=512,
            num_experts=3,
            temperature=1.0
        )
        
        x = torch.randn(2, 10, 512)
        expert_weights, router_logits = router(x)
        
        # Check shapes
        assert expert_weights.shape == (2, 10, 3)
        assert router_logits.shape == (2, 10, 3)
        
        # Check that weights sum to 1 along expert dimension
        weights_sum = expert_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)
        
        # Different tokens should have different routing (unlike pooled)
        # This is probabilistic, so we just check that not all are identical
        first_token_weights = expert_weights[:, 0, :]
        last_token_weights = expert_weights[:, -1, :]
        
        # At least some difference expected (with high probability)
        diff = torch.abs(first_token_weights - last_token_weights).sum()
        assert diff > 0.01  # Some difference expected
    
    def test_top_k_routing(self):
        """Test top-k sparse routing."""
        router = TokenLevelRouter(
            hidden_dim=512,
            num_experts=5,
            top_k=2  # Only use top 2 experts
        )
        
        x = torch.randn(2, 10, 512)
        expert_weights, router_logits = router(x)
        
        assert expert_weights.shape == (2, 10, 5)
        
        # Each token should have at most 2 non-zero expert weights
        for b in range(2):
            for t in range(10):
                non_zero_experts = (expert_weights[b, t, :] > 1e-6).sum()
                assert non_zero_experts <= 2
        
        # Weights should still sum to 1
        weights_sum = expert_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)
    
    def test_with_attention_mask(self):
        """Test token-level router with attention mask."""
        router = TokenLevelRouter(hidden_dim=512, num_experts=3)
        
        x = torch.randn(2, 10, 512)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
        
        expert_weights, router_logits = router(x, attention_mask)
        
        # Masked tokens should have zero weights
        assert (expert_weights[0, 3:, :] == 0).all()  # Tokens 3-9 should be masked
        assert (expert_weights[1, 5:, :] == 0).all()  # Tokens 5-9 should be masked
        
        # Non-masked tokens should have valid weights
        assert (expert_weights[0, :3, :].sum(dim=-1) - 1).abs().max() < 1e-5
        assert (expert_weights[1, :5, :].sum(dim=-1) - 1).abs().max() < 1e-5


class TestRouterFactory:
    """Test router factory function."""
    
    def test_create_simple_router(self):
        """Test creating simple router via factory."""
        router = create_router("simple", hidden_dim=512, num_experts=3)
        assert isinstance(router, SimpleRouter)
        assert router.hidden_dim == 512
        assert router.num_experts == 3
    
    def test_create_token_level_router(self):
        """Test creating token-level router via factory."""
        router = create_router("token", hidden_dim=512, num_experts=3)
        assert isinstance(router, TokenLevelRouter)
        assert router.hidden_dim == 512
        assert router.num_experts == 3
        
        # Test alternative name
        router2 = create_router("token_level", hidden_dim=512, num_experts=3)
        assert isinstance(router2, TokenLevelRouter)
    
    def test_invalid_router_type(self):
        """Test that invalid router type raises error."""
        with pytest.raises(ValueError):
            create_router("invalid_type", hidden_dim=512, num_experts=3)


class TestRouterMetrics:
    """Test router metrics and regularization functions."""
    
    def test_router_entropy(self):
        """Test router entropy computation."""
        # Uniform distribution should have high entropy
        uniform_logits = torch.zeros(2, 3)  # [batch, num_experts]
        uniform_entropy = compute_router_entropy(uniform_logits)
        
        # Peaked distribution should have low entropy  
        peaked_logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        peaked_entropy = compute_router_entropy(peaked_logits)
        
        assert uniform_entropy > peaked_entropy
        assert uniform_entropy > 0
        assert peaked_entropy >= 0
    
    def test_load_balancing_loss(self):
        """Test load balancing loss computation."""
        # Balanced usage should have low loss
        balanced_weights = torch.ones(2, 10, 3) / 3  # Equal weights for all experts
        balanced_loss = compute_load_balancing_loss(balanced_weights)
        
        # Imbalanced usage should have high loss
        imbalanced_weights = torch.zeros(2, 10, 3)
        imbalanced_weights[:, :, 0] = 1.0  # Only use first expert
        imbalanced_loss = compute_load_balancing_loss(imbalanced_weights)
        
        assert imbalanced_loss > balanced_loss
        assert balanced_loss >= 0
        assert imbalanced_loss >= 0
    
    def test_load_balancing_with_mask(self):
        """Test load balancing loss with attention mask."""
        weights = torch.ones(2, 10, 3) / 3
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
        
        loss_with_mask = compute_load_balancing_loss(weights, attention_mask)
        loss_without_mask = compute_load_balancing_loss(weights)
        
        # Both should be low for balanced usage
        assert loss_with_mask >= 0
        assert loss_without_mask >= 0


def test_router_training_mode():
    """Test that routers work in training mode."""
    router = SimpleRouter(hidden_dim=512, num_experts=3)
    router.train()
    
    x = torch.randn(2, 10, 512, requires_grad=True)
    expert_weights, router_logits = router(x)
    
    # Compute dummy loss and backward pass
    loss = expert_weights.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    for param in router.parameters():
        if param.requires_grad:
            assert param.grad is not None


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running router tests...")
    
    test_simple = TestSimpleRouter()
    test_simple.test_pooled_routing_mean()
    test_simple.test_attention_pooling()
    test_simple.test_with_attention_mask()
    print("✅ SimpleRouter tests passed")
    
    test_token = TestTokenLevelRouter()
    test_token.test_token_level_routing()
    test_token.test_top_k_routing()
    test_token.test_with_attention_mask()
    print("✅ TokenLevelRouter tests passed")
    
    test_factory = TestRouterFactory()
    test_factory.test_create_simple_router()
    test_factory.test_create_token_level_router()
    print("✅ Router factory tests passed")
    
    test_metrics = TestRouterMetrics()
    test_metrics.test_router_entropy()
    test_metrics.test_load_balancing_loss()
    test_metrics.test_load_balancing_with_mask()
    print("✅ Router metrics tests passed")
    
    test_router_training_mode()
    print("✅ Training mode tests passed")
    
    print("\n🎉 All router tests passed!")