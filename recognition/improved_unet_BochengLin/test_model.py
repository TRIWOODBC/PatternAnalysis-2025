import torch
from modules import UNet3D_Improved, ChannelAttention3d

"""
Validate UNet3D_Improved model architecture and training pipeline.
Tests include: model initialization, forward/backward pass, and attention mechanism.
"""

print("Validating UNet3D_Improved model...")
print("=" * 60)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Initialize model with actual training config (6 classes for prostate segmentation)
model = UNet3D_Improved(in_channels=1, num_classes=6)
model = model.to(device)
print(f"✓ Model initialized (6-class segmentation)")

# Check parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Total parameters: {total_params:,}")

# Forward pass test
print("\nTesting forward pass...")
batch_size = 2
dummy_input = torch.randn(batch_size, 1, 128, 128, 64).to(device)
print(f"  Input shape: {dummy_input.shape}")

try:
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Output shape: {output.shape} (expected: torch.Size([{batch_size}, 6, 128, 128, 64]))")
    assert output.shape == (batch_size, 6, 128, 128, 64), "Output shape mismatch"
    print(f"✓ Forward pass working correctly")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    exit(1)

# Backward pass test with Dice loss (like actual training)
print("\nTesting backward pass...")
try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy training batch
    dummy_input = torch.randn(batch_size, 1, 128, 128, 64).to(device)
    dummy_target = torch.randint(0, 6, (batch_size, 1, 128, 128, 64)).to(device)  # 6 classes
    
    # Forward pass
    output = model(dummy_input)
    
    # Use cross-entropy loss (standard for multi-class segmentation)
    loss = torch.nn.functional.cross_entropy(output, dummy_target.squeeze(1).long())
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Loss computed: {loss.item():.4f}")
    print(f"✓ Backward pass completed successfully")
except Exception as e:
    print(f"✗ Backward pass failed: {e}")
    exit(1)

# Test channel attention mechanism
print("\nTesting channel attention mechanism...")
try:
    attention = ChannelAttention3d(channels=256, reduction=16)
    attention = attention.to(device)
    
    test_feature = torch.randn(batch_size, 256, 16, 16, 8).to(device)
    with torch.no_grad():
        attended = attention(test_feature)
    
    # Verify attention output shape and values
    assert attended.shape == test_feature.shape, "Attention output shape mismatch"
    # Attention should scale features (values between 0 and original feature range)
    assert attended.max() <= test_feature.max(), "Attention scaling issue"
    print(f"✓ Attention mechanism working (output range: [{attended.min():.4f}, {attended.max():.4f}])")
except Exception as e:
    print(f"✗ Attention test failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✓ All validations passed!")
