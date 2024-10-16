import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#S - spot price
#K - strike price
#r - risk-free interest rate
#t - time to maturity
#sigma - volatility of stock

#Black Scholes Model Function
def blackScholes(S, K, T, r, sigma):

    #Calculate d1 and d2 from equation
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    #Calculate Call and Put prices using their respecitve formulas
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return call, put, d1, d2

#Calculating Volatility Function
def calcVolatility(prices):
    returns = np.diff(prices) / prices[:-1]  # Daily returns
    dailyVolatility = np.std(returns)

    #Annualizing the volatility (252 trading days in a year)
    yearlyVolatility = dailyVolatility * np.sqrt(252)

    return yearlyVolatility

#Retrieve the stock's data from Yahoo Finance on a range of certain days
def getStockData(ticker, daysFetch):

    #Set the variables using the yf module
    stock = yf.Ticker(ticker)
    endDate = datetime.now()
    startDate = endDate - timedelta(days=daysFetch)
    data = stock.history(start=startDate, end=endDate)

    #Get the trading days closing proces
    prices = data['Close'].dropna()

    return prices

#Visualize the past stock data using matplotlib
def getGraph(prices, volatility, ticker):
    days = np.arange(1, len(prices) + 1)

    #Create the plot and graph the data
    plt.figure(figsize=(10, 6))
    plt.plot(days, prices, label='Stock Closing Prices', color='green')
    plt.title(f'{ticker.upper()} Stock Prices Over Time')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Output the calculated volatility to the 5th decimal place
    print(f'Calculated volatility: {volatility:.5f}')

#Running the main program
def main():

    #Get the user's input for each variable and thier respective variable types
    stockTicker = input("Enter stock ticker:\n")
    K = float(input("Enter the strike price of the option to calculate:\n"))
    T = float(input("Enter the time to expiration in days:\n"))
    T /=365
    r = float(input("Enter the risk-free interest rate as a decimal (typically 0.05):\n"))
    daysFetch = int(input("Enter how many days you would like to analyze for volatility:\n"))

    try:
        # Fetch the requested number of trading days of closing prices
        prices = getStockData(stockTicker, daysFetch)
    except ValueError as e:
        print(e)
        return
    
    #Get the current stock price from the data fetched
    S = prices[-1] 
    sigma = calcVolatility(np.array(prices))

    #Calculate the options prices using the Black Scholes model
    call, put, d1, d2 = blackScholes(S,K,T,r,sigma)

    #Display the results from the calculations (round to 5th decimal place)
    print(f"\nStock Ticker: {stockTicker.upper()}")
    print(f"Stock Price (S): {S:.5f}")
    print(f"Strike Price (K): {K:.5f}")
    print(f"Time to Maturity (T): {T:.5f} years")
    print(f"Risk-Free Rate (r): {r}")
    print(f"Volatility (σ): {sigma:.5f} (annualized)")
    print(f"\nd1: {d1:.5f}")
    print(f"d2: {d2:.5f}")
    print(f"\nCall Option Price: {call:.5f}")
    print(f"Put Option Price: {put:.5f}")

    # Visualize stock data and volatility
    getGraph(prices, sigma, stockTicker)

#Run
if __name__ == "__main__":
    main()
