import yfinance as yf
import pandas as pd
from fredapi import Fred
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import r2_score


# API key (replace 'my key' with your FRED API key)
api_key = '63982cb073ceed99e4a225a778c00b71'
fred = Fred(api_key=api_key)

# List of indicators and their respective FRED tickers
indicators = {
    "Unemployment": "UNRATE",
    "Labor Force Participation": "CIVPART",
    "Nonfarm Employment": "PAYEMS",
    "Industrial Production": "INDPRO",
    "Retail Sales": "RETAIL",
    "Personal Consumption": "PCEPILFE",
    "Business Inventories": "BUSINV",
    "Housing Starts": "HOUST",
    "Corporate Profits": "CP",
    "Durable Goods Orders": "DGORDER",
    "Claims for Unemployment": "ICNSA",
    "Personal Consumption": "PCECC96",
    "Domestic Investment": "GPDI",
    "Consumer Sentiment": "UMCSENT",
    "Labor Productivity": "PRS85006092",
    "Delinquency on Mortgages": "DRSFRMACBS",
    "Delinquency on Commercial Loans": "DRBLACBS",
    "Total Vehicle Sales": "TOTALSA",
    "Delinquency on Credit Cards": "DRCCLACBS",
    "Profits After Tax": "CPATAX",
    "Construction Spending": "TTLCONS",
    "Real Disposable Personal Income": "DSPIC96",
    "Job Openings (Total Nonfarm)": "JTSJOL"
}


data = {}
for name, ticker in indicators.items():
    print(name)
    data[name] = fred.get_series(ticker, frequency='q')

# Convert dictionary to DataFrame
df_indicators_quarterly = pd.DataFrame(data)

# Calculate quarterly percentage change for the macroeconomic data
df_indicators_quarterly = df_indicators_quarterly.pct_change() * 100

# Download SPX prices
spx_data = yf.download('^GSPC', start='1970-01-01', end='2025-01-01')


# Resample SPX data to quarterly by taking the last close of each quarter
spx_data_quarterly = spx_data['Close'].resample('Q').last()

start_date_spx = spx_data_quarterly.index.min()  # Use the earliest date from SPX data
start_date_macro = df_indicators_quarterly.index.min()  # Use the earliest date from macro data

# Use the later of the two dates
start_date = max(start_date_spx, start_date_macro)

# Trim SPX data to the start date
spx_data_quarterly = spx_data_quarterly[spx_data_quarterly.index >= start_date]

# Optionally, if you want to trim the macro data too, you can do:
df_indicators_quarterly = df_indicators_quarterly[df_indicators_quarterly.index >= start_date]


# Reindex the SPX data to match the index of df_indicators_quarterly
spx_data_reindexed = spx_data_quarterly.reindex(df_indicators_quarterly.index)


# Fill NaN values in the reindexed spx_data using the nearest non-NaN value from the original spx_data
spx_data_reindexed = spx_data_reindexed.fillna(spx_data['Close'].reindex(df_indicators_quarterly.index, method='nearest'))


# Merge SPX with our quarterly data
df_merged = df_indicators_quarterly.merge(spx_data_reindexed, left_index=True, right_index=True, how='left')

original_df_merged = df_merged.copy()


for y in range(1, 3):
    print(f'Range Iteration no. {y}')
    df_merged['Close Chg'] = original_df_merged['Close'].pct_change() * 100

    
    for x in range(1,2):
        df_merged['Shifted Close'] = df_merged['Close Chg'].shift(x)
        r2_values = {}
        df_merged = df_merged.dropna()
        
        # Generate scatter plots for all indicators
        fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(20, 20), constrained_layout=True)
        
        indicators_list = list(indicators.keys())
        
        all_r2_values = {}
        for i, ax in enumerate(axes.ravel()):
            if i < len(indicators_list):
                indicator = indicators_list[i]
                
                # Use the 'Shifted Close' column for plotting
                valid_data = df_merged[[indicator, 'Shifted Close']]
                ax.scatter(valid_data[indicator], valid_data['Shifted Close'], alpha=0.6, edgecolors="k", linewidths=0.5)
                print(valid_data)
                
                # Add a linear regression line
                try:
                    m, b = np.polyfit(valid_data[indicator], valid_data['Shifted Close'], 1)
                    predicted = m * valid_data[indicator] + b
                    ax.plot(valid_data[indicator], predicted, color='red')
                    
                    # Calculate R-squared
                    r2 = r2_score(valid_data['Shifted Close'], predicted)
                    r2_values[indicator] = r2
                    
                except np.linalg.LinAlgError:
                    print(f"Couldn't fit data for indicator: {indicator}")
                    r2_values[indicator] = None  # or some default value
                
                ax.set_title(indicator)
                ax.set_xlabel('Quarterly Indicator Value')
                ax.set_ylabel(f'SPX Quarterly Returns')
                ax.grid(True, which="both", ls="--", c='0.65')
    
        all_r2_values[x] = r2_values
        avg = sum(filter(None, r2_values.values())) / len(r2_values)
        
        plt.tight_layout()
        plt.show()

        avg_value_dict = {}
        
        for shift, r2s in all_r2_values.items():
            print(f"Shift: {shift}, Period: {y}")
            total_r2 = 0
            valid_r2_count = 0
            for indicator, r2 in r2s.items():
                print(f"R-squared for {indicator}: {round(r2, 2) if r2 is not None else None}")
    
                if r2 is not None:   # Add this condition to make sure you're not adding None values
                    total_r2 += r2
                    valid_r2_count += 1
                
            average_r2 = total_r2 / valid_r2_count
            print(f"Average R-squared for shift {shift}: {round(average_r2, 2) if average_r2 is not None else None}\n")
            avg_value_dict[f'shift: {shift}, period: {y}'] = average_r2

        for key, value in avg_value_dict.items():
            print(f'{key} - {value}')
