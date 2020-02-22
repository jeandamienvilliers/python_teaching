# -*- coding: utf-8 -*-
"""
@author: jeand
"""
#Importing packages
import os 
import numpy as np
import pandas as pd
from matplotlib.pyplot import scatter
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize
import json
from scipy.optimize import minimize


# Edit this line with the folder contain raw data files (.txt, .csv, .json...)
raw_data_folder=r"XXX\raw data" 



#Exercise 2.1
print("Exercise 2.1")
def randlist(n,a,b):    
    main_list=[np.random.randint(a,b) for i in range(n)]
    return main_list
print(f"the result of randlist(10,25,50) is {randlist(10,25,50)}")





#Exercise 2.2
print("Exercise 2.2")
def filter_liquid_options(csv_file):
    df=pd.read_csv(raw_data_folder+"\\"+csv_file) #Load the csv file
    for column_header in ["Bid", "Ask", "Volume"]: # Loops on the Bid/Ask/Volume columns
        df[column_header]=[float(x.replace(",","")) for x in df[column_header]] # Converts  column to float
    df["Mid"]=(df["Bid"]+df["Ask"])/2.0 #Computes the mid
    df=df[df["Volume"]!=0] # Filter out option with 0 volume
    return df

df_Liquid_SP500Options=filter_liquid_options("Option_SP500.csv")
print(f"The result of filter_liquid_options(Option_SP500.csv) is \n \n : {df_Liquid_SP500Options.head(5)} (\n only 5 first rows displayed)")


#Exercise 2.3
print("Exercise 2.3")
#The function clean prices takes future as input. It reads the correspondon CSV files, converts them into a pandas dataframe, 
# and builds a new dataframe with the spot prices of dates common to those future futures
def get_clean_prices(list_futures):
    dict_futures={}
    list_dates=[]
    df_new=pd.DataFrame()
    for Future in list_futures: #loops on each future
        temporary_df=pd.read_csv(raw_data_folder+"\\"+Future+".csv") # reads the csv file
        dict_futures[Future]=temporary_df # store the dataframe in a dictionary
        list_dates.append(list(temporary_df["Date"])) #Store the list of dates of the dataframes
    
    list_common_dates=set(list_dates[0])
    for s in list_dates[1:]:
        list_common_dates.intersection_update(s) # finds the dates commn to all futures   
    for Future in dict_futures.keys():    
        dict_futures[Future]=dict_futures[Future][dict_futures[Future]["Date"].isin(list_common_dates)] # Keep only common dates       
        prices=list(dict_futures[Future]["Settle"])#list of settlement prices
        df_new[Future]=prices
    df_new.index=dict_futures[list_futures[0]]["Date"]
    return df_new

list_futures=["CAC_FUTURE", "FTSE100_FUTURE", "MSCI_EUROPE_FUTURE"]
clean_prices=get_clean_prices(list_futures)
print(f"The result of get_clean_prices({list_futures}) is \n \n : {clean_prices.head(5)} ( \n only 5 first rows displayed)")

#The function compute_correl_matrix calls the get_clean_prices to clean the data, then computes Future daily returns
# and computes with Numpy the correlation matrix using corrcoef function
def compute_correl_matrix(list_futures):
    clean_prices=get_clean_prices(list_futures)
    list_returns=[]
    for future in list_futures:
        prices=clean_prices[future]
        returns=[prices[i]/prices[i-1]-1 for i in range(1,len(prices))]
        list_returns.append(returns)

    correl_matrix=np.corrcoef(list_returns)
    return correl_matrix

list_futures=["CAC_FUTURE", "FTSE100_FUTURE", "MSCI_EUROPE_FUTURE"]
correl_matrix=compute_correl_matrix(list_futures)
print(f"The result of correl_matrix({list_futures}) is \n \n : {correl_matrix} ( \n \n only 5 first rows displayed)")




#Exercise 2.4
print("Exercise 2.4")
def generate_bivariate_normal(means, stds, corr, N): 
    covs = [[stds[0]**2          , stds[0]*stds[1]*corr], # Covariance matrix
            [stds[0]*stds[1]*corr,           stds[1]**2]]  
    
    m = np.random.multivariate_normal(means, covs, N).T # Generate 2 x N random gaussian variables as an array and transpose the matrix (.T method) 
    scatter(m[0], m[1]) # Plot the generated random variable
generate_bivariate_normal([0,0], [0.2, 0.1], 0.9, 1000)






#Exercise 2.5
print("Exercise 2.5")
mu, sigma = 0, 1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000) # generates 1000 random gaussian variables of mean mu / std sigma
observed_mean=np.mean(s) # computes empirical mean (from the generated sample)
observed_std=np.std(s) # computes std (from generated sample)

def gaussian_distribution(mu, sigma): # Density of the gaussian distribution
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) )

count, bins, ignored = plt.hist(s, 100, density=True) # Plot the empirical distribution with 100 bins
plt.plot(bins, gaussian_distribution(mu, sigma), # plot the theoretical distribution in red
          linewidth=2, color='r')
plt.show() # show the graph



#Exercise 2.6
print("Exercise 2.6")
# Function to compute the linear regression between the returns of two futures
def compute_linear_regression(list_futures):
    clean_prices=get_clean_prices(list_futures)
    spot_x=clean_prices[list_futures[0]]
    returns_x=[spot_x[i]/spot_x[i-1]-1 for i in range(1,len(spot_x))]
    spot_y=clean_prices[list_futures[1]]
    returns_y=[spot_y[i]/spot_y[i-1]-1 for i in range(1,len(spot_y))]
    slope, intercept, r_value, p_value, std_err = stats.linregress(returns_x, returns_y)
    print("slope, intercept, r_value, p_value, std_err")
    print(slope, intercept, r_value, p_value, std_err)   
    plt.plot(returns_x, returns_y, 'o', label='original data')
    plt.plot(returns_x, intercept + slope*np.array(returns_x), 'r', label='fitted line')
    plt.legend()
    plt.show()
list_futures=["CAC_FUTURE", "FTSE100_FUTURE"]
compute_linear_regression(list_futures)


#Exercise 2.7
print("Exercise 2.7")
#Function to make a pandas pivot table
def make_pivot_table(csv_file):
    df=pd.read_csv(raw_data_folder+"\\"+csv_file) #Load the csv file
    column_header="Volume"
    df[column_header]=[float(x.replace(",","")) for x in df[column_header]] # Converts  column to float
    df_pivot=pd.pivot_table(data=df, index=["Maturity"], columns=["Option Type"], values=["Volume"], aggfunc=np.sum)
    return df_pivot
pt=make_pivot_table("Option_SP500.csv")
print(f"The result of make_pivot_table(Option_SP500.csv) is \n \n : {pt} ")



#Exercise 2.8
print("Exercise 2.8")
#Function to read a json file
def get_json_file_name(json_file):
    jsonfile_read = open(raw_data_folder+"\\"+json_file, "r") # Open the json file
    decoded_json = json.load(jsonfile_read) # Load the json file as python dictionary
    name=decoded_json.get("dataset")["name"] # Read content of the dictionary
    return name
json_file="FEXDZ2014.json"
json_name=get_json_file_name(json_file)
print(f"The result of get_json_file_name({json_file}) is \n \n : {json_name} ")



#Exercise 2.9
print("Exercise 2.9")
#Function using Scipy.Optimize to find the minimum of a function
def find_minimum(g, x0):
    res = minimize(g, [x0], tol=1e-6) # call scipy with tolerange threshold at 10^{-6}
    result=res.x[0]
    return result

g = lambda x: 0.5 * np.exp(-x * (1-x))
x0=0
g_minimum=find_minimum(g, x0)
print(f"The result of find_minimum({g}, {x0}) is \n \n : {g_minimum} ")




