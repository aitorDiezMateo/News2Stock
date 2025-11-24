"""
Test script to verify Pointer-Generator Network installation
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found. Install with: pip install torch")
        return False
    
    try:
        import pandas
        print(f"✓ Pandas {pandas.__version__}")
    except ImportError:
        print("✗ Pandas not found. Install with: pip install pandas")
        return False
    
    try:
        import pyarrow
        print(f"✓ PyArrow {pyarrow.__version__}")
    except ImportError:
        print("✗ PyArrow not found. Install with: pip install pyarrow")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not found. Install with: pip install numpy")
        return False
    
    return True


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available (will use CPU)")
        print("  Training will be slower but still work")
    
    return True


def test_modules():
    """Test if local modules can be imported"""
    print("\nTesting local modules...")
    
    try:
        import config
        print("✓ config.py")
    except ImportError as e:
        print(f"✗ config.py: {e}")
        return False
    
    try:
        import vocabulary
        print("✓ vocabulary.py")
    except ImportError as e:
        print(f"✗ vocabulary.py: {e}")
        return False
    
    try:
        import model
        print("✓ model.py")
    except ImportError as e:
        print(f"✗ model.py: {e}")
        return False
    
    try:
        import dataset
        print("✓ dataset.py")
    except ImportError as e:
        print(f"✗ dataset.py: {e}")
        return False
    
    try:
        import preprocess
        print("✓ preprocess.py")
    except ImportError as e:
        print(f"✗ preprocess.py: {e}")
        return False
    
    return True


def test_data():
    """Test if data file exists"""
    print("\nTesting data availability...")
    
    import config
    
    if os.path.exists(config.DATA_PATH):
        print(f"✓ Data file found: {config.DATA_PATH}")
        
        # Check file size
        size_mb = os.path.getsize(config.DATA_PATH) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
        
        return True
    else:
        print(f"✗ Data file not found: {config.DATA_PATH}")
        print("  Make sure the data file exists before preprocessing")
        return False


def test_model_creation():
    """Test if model can be created"""
    print("\nTesting model creation...")
    
    try:
        import torch
        import config
        from model import Encoder, Decoder, Attention, PointerGeneratorNetwork
        
        device = torch.device('cpu')  # Use CPU for testing
        
        # Create small model for testing
        encoder = Encoder(
            vocab_size=1000,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=1,
            dropout=0.3
        )
        
        attention = Attention(
            encoder_hidden_dim=256,
            decoder_hidden_dim=128,
            use_coverage=True
        )
        
        decoder = Decoder(
            vocab_size=1000,
            embedding_dim=64,
            encoder_hidden_dim=256,
            decoder_hidden_dim=128,
            num_layers=1,
            dropout=0.3,
            attention=attention
        )
        
        model = PointerGeneratorNetwork(encoder, decoder, device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created successfully")
        print(f"  Parameters: {num_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False


def test_forward_pass():
    """Test if forward pass works"""
    print("\nTesting forward pass...")
    
    try:
        import torch
        import config
        from model import Encoder, Decoder, Attention, PointerGeneratorNetwork
        
        device = torch.device('cpu')
        
        # Create small model
        encoder = Encoder(1000, 64, 128, 1, 0.3)
        attention = Attention(256, 128, use_coverage=True)
        decoder = Decoder(1000, 64, 256, 128, 1, 0.3, attention)
        model = PointerGeneratorNetwork(encoder, decoder, device)
        
        # Create dummy input
        batch_size = 2
        src_len = 10
        trg_len = 5
        
        src = torch.randint(0, 1000, (batch_size, src_len))
        src_extended = torch.randint(0, 1010, (batch_size, src_len))  # With 10 OOV words
        src_lengths = torch.tensor([src_len, src_len])
        trg = torch.randint(0, 1000, (batch_size, trg_len))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            final_dists, coverages = model(src, src_lengths, trg, src_extended, oov_size=10, teacher_forcing_ratio=0)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {final_dists.shape}")
        print(f"  Coverage steps: {len(coverages)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

