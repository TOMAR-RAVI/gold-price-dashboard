<<<<<<< HEAD
import yfinance as yf
import pandas as pd

# Gold Futures ticker symbol on Yahoo Finance
ticker = 'GC=F'

# Download gold price data from 2010 to 2025
gold_data = yf.download(ticker, start='2010-01-01', end='2025-01-01')

# Save the data to a CSV file
gold_data.to_csv('gold_prices.csv')

# Print the first few rows to check
print(gold_data.head())
=======
import yfinance as yf
import pandas as pd

# Gold Futures ticker symbol on Yahoo Finance
ticker = 'GC=F'

# Download gold price data from 2010 to 2025
gold_data = yf.download(ticker, start='2010-01-01', end='2025-01-01')

# Save the data to a CSV file
gold_data.to_csv('gold_prices.csv')

# Print the first few rows to check
print(gold_data.head())
>>>>>>> 31bd6b5b60ec2a60820fc1d7e94d63d2adc2eb94
