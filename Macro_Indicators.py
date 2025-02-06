import yfinance as yf
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import r2_score


api_key = 'here'
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


# Fetch data and store in a dictionary
data = {}
for name, ticker in indicators.items():
    print(name)
    data[name] = fred.get_series(ticker, frequency='q')

# Convert dictionary to DataFrame
df_indicators_quarterly = pd.DataFrame(data)
    
df_indicators_quarterly = df_indicators_quarterly.pct_change() * 100

# Download SPX prices
spx_data = yf.download('^GSPC', start='1970-01-01', end='2023-09-24')

# Reindex spx_data to match the index of df_indicators_quarterly
spx_data_reindexed = spx_data.reindex(df_indicators_quarterly.index)

# Fill NaN values in the reindexed spx_data using the nearest non-NaN value from the original spx_data
spx_data_reindexed['Close'] = spx_data_reindexed['Close'].fillna(spx_data['Close'].reindex(df_indicators_quarterly.index, method='nearest'))

# Merge SPX with our quarterly data
df_merged = df_indicators_quarterly.merge(spx_data_reindexed['Close'], left_index=True, right_index=True, how='left')

original_df_merged = df_merged.copy()

for y in range(1, 4):
    print(f'Range Iteration no. {y}')
    df_merged['Close Chg'] = original_df_merged['Close'].pct_change() * 100
    
    for x in range(-4,4):
        df_merged['Shifted Close'] = df_merged['Close Chg'].shift(x)
        r2_values = {}
        
        # Generate scatter plots for all indicators
        # right now I dont want to generate charts as they take a lot of time
        #fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(20, 20), constrained_layout=True)
        
        indicators_list = list(indicators.keys())
        
        all_r2_values = {}
        for i, indicator in enumerate(indicators_list): #if you want to see charts change indicator to ax and indicators list to axes.ravel()
            if i < len(indicators_list):
                indicator = indicators_list[i]
                
                # Use the 'Shifted Close' column for plotting
                valid_data = df_merged[[indicator, 'Shifted Close']].replace([np.inf, -np.inf], np.nan).dropna()
                #ax.scatter(valid_data[indicator], valid_data['Shifted Close'], alpha=0.6, edgecolors="k", linewidths=0.5)
                
                # Add a linear regression line
                try:
                    m, b = np.polyfit(valid_data[indicator], valid_data['Shifted Close'], 1)
                    predicted = m * valid_data[indicator] + b
                    #ax.plot(valid_data[indicator], predicted, color='red')
                    
                    # Calculate R-squared
                    r2 = r2_score(valid_data['Shifted Close'], predicted)
                    r2_values[indicator] = r2
                    
                except np.linalg.LinAlgError:
                    print(f"Couldn't fit data for indicator: {indicator}")
                    r2_values[indicator] = None  # or some default value
                
                #ax.set_title(indicator)
                #ax.set_xlabel('Quarterly Indicator Value')
                #ax.set_ylabel(f'SPX Quarterly Returns')
                #ax.grid(True, which="both", ls="--", c='0.65')
    
        all_r2_values[x] = r2_values
        avg = sum(filter(None, r2_values.values())) / len(r2_values)
        
        #plt.tight_layout()
        #plt.show()

        avg_value_dict = {}
        
        for shift, r2s in all_r2_values.items():
            print(f"Shift (Quarters): {shift}, Period (Quarters): {y}")
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
