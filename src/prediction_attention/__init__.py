"""
Attention-based Stock Price Movement Prediction Module
======================================================
Uses daily news aggregation + Self-Attention mechanism.

Strategy:
1. Aggregate news embeddings per day (mean within each day)
2. Create sequence: (window_size, embedding_dim) - one vector per day
3. Use Attention mechanism to learn which days are important
4. Combine with stock embeddings for final prediction
"""
