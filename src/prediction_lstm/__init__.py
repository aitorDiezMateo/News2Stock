"""
LSTM-based Stock Price Movement Prediction Module
=================================================
Uses daily news aggregation + LSTM/GRU to process temporal sequences.

Strategy:
1. Aggregate news embeddings per day (mean within each day)
2. Create sequence: (window_size, embedding_dim) - one vector per day
3. Use LSTM/GRU to process the temporal sequence
4. Combine with stock embeddings for final prediction
"""
