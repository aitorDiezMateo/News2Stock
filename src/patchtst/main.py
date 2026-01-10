"""
Main Training Script for PatchTST MLM
======================================
Pre-train PatchTST on stock data using masked language modeling.
Uses 20-day windows to learn robust time series representations.

Usage:
    python train.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader

from patchtst import (
    Config,
    PatchTST_MLM,
    StockWindowDataset,
    create_train_val_split,
    set_seed,
    get_lr_scheduler,
    train_epoch,
    validate,
    save_checkpoint,
    load_checkpoint,
    extract_embeddings,
    save_embeddings_by_ticker,
    plot_training_history,
    plot_learning_rate,
    plot_loss_distribution,
)


def main():
    """Main training loop."""
    
    # Print header
    print("\n" + "="*70)
    print("PATCHTST MLM PRE-TRAINING ON STOCK DATA")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(Config.SEED)
    
    # Print configuration
    Config.print_config()
    
    # Create output directories
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.EMBEDDINGS_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.PLOTS_PATH, exist_ok=True)
    
    # ========================================================================
    # 1. CREATE DATASET
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Creating Dataset")
    print("="*70)
    
    dataset = StockWindowDataset(
        tickers=Config.TICKERS,
        window_size=Config.WINDOW_SIZE,
        target_col=Config.TARGET_COL,
        data_path=Config.DATA_PATH_LOAD,
        stride=Config.STRIDE
    )
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  - Mean: {stats['mean']:.6f}")
    print(f"  - Std: {stats['std']:.6f}")
    print(f"  - Min: {stats['min']:.6f}")
    print(f"  - Max: {stats['max']:.6f}")
    
    # Split into train/val
    train_dataset, val_dataset = create_train_val_split(
        dataset, 
        train_ratio=Config.TRAIN_SPLIT,
        seed=Config.SEED
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # ========================================================================
    # 2. CREATE MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Creating Model")
    print("="*70)
    
    model = PatchTST_MLM(
        seq_len=Config.WINDOW_SIZE,
        patch_size=Config.PATCH_SIZE,
        stride=Config.PATCH_STRIDE,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
        mask_ratio=Config.MASK_RATIO,
        mask_type=Config.MASK_TYPE,
        use_revin=Config.USE_REVIN
    )
    
    model = model.to(Config.DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Trainable parameters: {num_trainable:,}")
    print(f"  - Number of patches: {model.num_patches}")
    print(f"  - Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB")
    
    # ========================================================================
    # 3. CREATE OPTIMIZER AND SCHEDULER
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Setting up Optimizer")
    print("="*70)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = get_lr_scheduler(
        optimizer, 
        Config.WARMUP_EPOCHS, 
        Config.NUM_EPOCHS
    )
    
    print(f"\nOptimizer: AdamW")
    print(f"  - Learning rate: {Config.LEARNING_RATE}")
    print(f"  - Weight decay: {Config.WEIGHT_DECAY}")
    print(f"  - Warmup epochs: {Config.WARMUP_EPOCHS}")
    
    # ========================================================================
    # 4. TRAINING LOOP
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: Training")
    print("="*70)
    
    train_losses = []
    val_losses = []
    lr_history = []
    best_val_loss = float('inf')
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"{'─'*70}")
        
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        print(f"Learning rate: {current_lr:.6f}")
        
        # Train
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            Config.DEVICE,
            gradient_clip=Config.GRADIENT_CLIP
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, Config.DEVICE)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improvement = "✓ NEW BEST!"
            
            # Create a safe config dict
            config_dict = {}
            for k, v in Config.__dict__.items():
                if k.startswith('__'):
                    continue
                if isinstance(v, (int, float, str, bool, list, tuple, dict)):
                    config_dict[k] = v
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                filepath=f"{Config.MODEL_SAVE_PATH}best_model.pt",
                config=config_dict
            )
        else:
            improvement = ""
        
        print(f"  Best Val:   {best_val_loss:.6f} {improvement}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                filepath=f"{Config.MODEL_SAVE_PATH}checkpoint_epoch_{epoch+1}.pt"
            )
            print(f"  ✓ Checkpoint saved")
    
    # ========================================================================
    # 5. PLOT TRAINING HISTORY
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: Saving Training Visualizations")
    print("="*70)
    
    plot_training_history(
        train_losses, 
        val_losses, 
        save_path=f"{Config.PLOTS_PATH}training_history.png"
    )
    
    plot_learning_rate(
        lr_history,
        save_path=f"{Config.PLOTS_PATH}learning_rate.png"
    )
    
    plot_loss_distribution(
        train_losses,
        val_losses,
        save_path=f"{Config.PLOTS_PATH}loss_distribution.png"
    )
    
    # ========================================================================
    # 6. EXTRACT EMBEDDINGS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: Extracting Embeddings")
    print("="*70)
    
    # Load best model
    print("\nLoading best model...")
    checkpoint = load_checkpoint(
        f"{Config.MODEL_SAVE_PATH}best_model.pt",
        model
    )
    print(f"  - Loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"  - Best val loss: {checkpoint['val_loss']:.6f}")
    
    # Extract embeddings
    print("\nExtracting embeddings from all windows...")
    embeddings_dict = extract_embeddings(
        model=model,
        dataset=dataset,
        device=Config.DEVICE,
        pooling=Config.POOLING_STRATEGY,
        batch_size=Config.EMBEDDING_BATCH_SIZE
    )
    
    print(f"\nExtracted embeddings:")
    print(f"  - Shape: {embeddings_dict['embeddings'].shape}")
    print(f"  - Pooling: {Config.POOLING_STRATEGY}")
    
    # Save embeddings by ticker
    save_embeddings_by_ticker(
        embeddings_dict=embeddings_dict,
        save_path=Config.EMBEDDINGS_SAVE_PATH,
        window_size=Config.WINDOW_SIZE,
        target_col=Config.TARGET_COL
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\nFinal Results:")
    print(f"  - Best validation loss: {best_val_loss:.6f}")
    print(f"  - Final train loss: {train_losses[-1]:.6f}")
    print(f"  - Final val loss: {val_losses[-1]:.6f}")
    
    print(f"\nSaved Files:")
    print(f"  - Model: {Config.MODEL_SAVE_PATH}best_model.pt")
    print(f"  - Embeddings: {Config.EMBEDDINGS_SAVE_PATH}")
    print(f"  - Plots: {Config.PLOTS_PATH}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
