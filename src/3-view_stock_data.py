import matplotlib.pyplot as plt
import pandas as pd

data_google = pd.read_parquet('data_stocks/processed/GOOGL_data_processed.parquet')
plt.plot(data_google['Close'], label='GOOGL Close Price')
plt.show()