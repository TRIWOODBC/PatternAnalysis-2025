import torch
from modules import UNet3D_Improved

print("Testing UNet3D_Improved model...")
print("=" * 50)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model
model = UNet3D_Improved(in_channels=1, num_classes=2)
model = model.to(device)
print(f"✓ Model created")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Total parameters: {total_params:,}")

# Test forward pass
print("\nTesting forward pass...")
batch_size = 2
dummy_input = torch.randn(batch_size, 1, 128, 128, 64).to(device)
print(f"  Input shape: {dummy_input.shape}")

try:
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Forward pass successful!")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test backward pass
print("\nTesting backward pass...")
try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    dummy_input = torch.randn(batch_size, 1, 128, 128, 64, requires_grad=True).to(device)
    dummy_target = torch.randint(0, 2, (batch_size, 128, 128, 64)).to(device)
    
    output = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(output, dummy_target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Backward pass successful!")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("\n" + "=" * 50)
print("✓ All tests passed!")
