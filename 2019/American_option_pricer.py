'''
American Option pricer
Binomial Method
'''


#Import packages
from math import exp, sqrt


# Option parameters 
dt = float(input("Enter the timestep in days: "))/252.0 #ex:1
S = float(input("Enter the initial asset price: ")) #ex:100
r = float(input("Enter the risk-free discount rate: ")) #ex:0.01
K = float(input("Enter the option strike price: ")) #ex:100
p = float(input("Enter the asset growth probability p: ")) #ex:0.2
vol = float(input("Enter the volatility: ")) #ex:0.3
N = int(input("Enter the number of timesteps until expiration: ")) # ex:252
call = input("Is this a call or put option? (C/P) ") #ex:call

u=math.exp(vol*math.sqrt(dt)) # When spot goes up in the american tree we have: S(t+1)=u * S(t)


def spot_price(t, n_up):
    """ Compute the stock price after 'n_up' growths and 't - n_up' decays. """
    return S * (u ** (2 * n_up - t))

def binomial_price_recursive(t, n_up): #Main pricing function using a recursive algorithm   
    stockPrice = spot_price(t, n_up) # Get the spot price given the time and the number of growths
    if call=="call": exerciseProfit = max(0, stockPrice - K) # computes the intrinsic value
    else:    exerciseProfit = max(0, K - stockPrice)
    if t == N: 
        return exerciseProfit # If T=Maturity the function returns the intrinsic value
    else:

        ZC = exp(-r * dt) #Discount factor
        expected = p * binomial_price_recursive(t + 1, n_up + 1) + (1 - p) * binomial_price_recursive(t + 1, n_up) # Price expectation at time t+1
        binomial = ZC * expected # Dicounted price expectation
        return max(binomial, exerciseProfit) # Maximum between the intrinsic value and the expected discounted price at t+1

print(f"The price of the american option is: \n {binomial_price_recursive(0,0)}")
