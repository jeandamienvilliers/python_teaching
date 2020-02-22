# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:13:22 2019

@author: jeand
"""

#Import packages
import json
import datetime
from scipy import interpolate
import math
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy


# Edit this line with the folder contain raw data files (.txt, .csv, .json...)
raw_data_folder=r"C:\Users\jeand\Desktop\Python course\Python exercise\raw data" 



def iso_string_date_to_pythondate(isodate): # This function converts dates in iso format "YYYMMDD" into a python format
    return datetime.datetime(int(isodate[0:4]),int(isodate[4:6]),int(isodate[6:8]),12,00)




####################################################
############# Parsing Dividend Futures data ########
####################################################    
def parse_json_div_futures(div_code, start_year, end_year): # This function will read the various .json files and will return a python dictionary as output
    dict_all_div_dut_data={} # create empty dictionary
    for year in range(start_year,end_year+1): # Loop on each Div Future year
        year_str=str(year) # Convert number to string
        jsonfile = open(raw_data_folder+"\\"+div_code+year_str+'.json', "r") # Open the .json file
        div_fut_data = json.load(jsonfile) # Load the json file as python dictionary
        date_column_index=div_fut_data["dataset"]["column_names"].index("Date") # Extract list of dates index
        settle_column_index=div_fut_data["dataset"]["column_names"].index("Settle") # Extract list of settlement prices index
        for data in div_fut_data["dataset"]["data"]: # Loops on each item of the list "data"
            date=data[date_column_index] # Extract the date
            price=data[settle_column_index] # Extract the settlement price
            observation_date=iso_string_date_to_pythondate(date.replace("-","")) # Extracts the observation date and converts it to python date
            fut_matu_date=iso_string_date_to_pythondate(year_str+"1231") # Extracts the Div Fut maturity and converts it to python date
            ttm_days=fut_matu_date-observation_date # COmputes the time to maturity
            ttm_years=float(ttm_days.days)/365.0 # Converts time to maturity in years
            dict_all_div_dut_data[fut_matu_date, observation_date,ttm_years]=price #Store the information in the main python dictionary           
    return dict_all_div_dut_data # returns the main python dictionary
        
    
dict_futures_converging=parse_json_div_futures("FEXDZ", 2014, 2025)  # Executes the function parse_json_div_futures


def converging_to_sliding(dict_futures_converging): # This function converts the historical time serie of converging prices (2014/2015...) into sliding prices (1Y, 2Y...)
    list_observation_dates=list(set([x[1] for x in dict_futures_converging.keys()])) # Extract list of observation dates (list(set) to remove duplicates)
    dict_sliding_matu_prices={} # create empty dictionary
    for observation_date in list_observation_dates: # loops on each observation date
        sub_dict=dict([(k,v) for k,v in dict_futures_converging.items() if k[1]==observation_date ]) # create a sub dictionary with only data corresponding to the current observation date
        if len(sub_dict)>1: # Only if the number of prices is above 1
            list_fut_ttm=[x[2] for x in sub_dict.keys()] # Extract the list of time to maturities
            list_prices=list(sub_dict.values()) # Extract list of prices
            list_sliding_maturities=range(int(math.ceil(min(list_fut_ttm))),int(math.floor(max(list_fut_ttm)))) # Sliding maturities will be populated only between and the min and max time to maturity
            f = interpolate.interp1d(list_fut_ttm, list_prices) # Define a linear interpolation function using scipy package
            list_interpolated_prices=[float(f(x)) for x in list_sliding_maturities] # Apply the interpolation on the sliding maturities
            for i in range(len(list_sliding_maturities)): # Loops on the number of sliding maturities
                sliding_maturity=list_sliding_maturities[i] # Extract the sliding maturity   
                dict_sliding_matu_prices[sliding_maturity,observation_date]=list_interpolated_prices[i] # Stores the interpolated price in the dictionary
    df_futures_sliding=pd.DataFrame.from_dict(dict_sliding_matu_prices, orient="index") # COnvert the python dictionary into a dataframe
    df_futures_sliding.columns=["div_fut_price"] # Rename the column of the dataframe
    df_futures_sliding["sliding_maturity"]=[x[0] for x in df_futures_sliding.index] # Stores the Dataframe's index information in a dedicated dataframe column
    df_futures_sliding["observation_date"]=[x[1] for x in df_futures_sliding.index] # Stores the Dataframe's index information in a dedicated dataframe column
    return df_futures_sliding

df_futures_sliding=converging_to_sliding(dict_futures_converging) # Executes the function converging_to_sliding

####################################################
############# Parsing SX5E Spot data ###############
#################################################### 
def get_SX5E_spot_df(csv_file): # function to read a csv and store the historical close price of SX5E
    df_STOXX50=pd.read_csv(raw_data_folder+"\\"+csv_file) # reads the csv and stores it as a dataframe
    df_STOXX50["observation_date"]=[iso_string_date_to_pythondate(x.replace("-", "")) for x in df_STOXX50["Date"]] # COnverts dates to python format
    #df_STOXX50=df_STOXX50[df_STOXX50["Close"]!='null'] #  line commented, to be ignored in Python 3.7
    df_STOXX50=df_STOXX50.dropna() # Removes the rows with NA figures
    df_STOXX50["Close"]=[float(x) for x in df_STOXX50["Close"]] # COnverts the "Close" to floats
    return df_STOXX50 # returns the dataframe

df_STOXX50=get_SX5E_spot_df("STOXX50E.csv") # call the function get_SX5E_spot_df on a specific SX5E file





#########################################################################################
############# Linear regression between Div Futures and Equity spot moves ###############
############# for one sliding maturity ##################################################
#########################################################################################

'''
In this section we have as a time series of spot returns [dS / S] and a time series of div futures returns [d Div / Div].
We are going to perform a linear regression between both to get the dividend slope i.e.
dDiv / Div = Slope * (dS / S) + epsilon
'''


def compute_linear_regression_spot_div(df_STOXX50, df_futures_sliding, sliding_maturity):
    df_fut_historic=df_futures_sliding[df_futures_sliding["sliding_maturity"]==sliding_maturity] # Creates a sub dataframe containing only div futures prices of the requested sliding maturity
    df_merge= pd.merge(df_STOXX50, df_fut_historic, on=['observation_date']) # Merges Spot and Div Futures dataframes on the rows with identical observation_date
    list_sx5e_return=np.diff(np.log(df_merge["Close"])) # Computes SX5E Spot log returns
    list_sx5e_div_fut_return=np.diff(np.log(df_merge["div_fut_price"])) # COmputes Div Futures log returns
    slope, intercept, r_value, p_value, std_err = stats.linregress(list_sx5e_return,list_sx5e_div_fut_return) # Performs the linear regression between the log returns time series
    return slope, intercept, list_sx5e_return, list_sx5e_div_fut_return # return the slope, the intercept of the regression, and both time series


slope=compute_linear_regression_spot_div(df_STOXX50, df_futures_sliding, 8)[0] # Example calling compute_linear_regression_spot_div looking at the slope of the regression on maturity = 8 years

def plot_linear_regression(sliding_maturity,df_STOXX50,df_futures_sliding):  # This function is a scatter plot using matplotlib in order to visualize the SX5E / Div Futures returns vs the result of the linear regression   
    linreg=compute_linear_regression_spot_div(df_STOXX50, df_futures_sliding, sliding_maturity)
    slope=linreg[0]
    intercept=linreg[1]
    sx5e_returns=linreg[2]
    sx5e_divfut_returns=linreg[3]
    f_regression=lambda x:slope*x+intercept
    fig, axes = plt.subplots() #create plot object    
    axes.plot(sx5e_returns, sx5e_divfut_returns, '.') #plot the XY scatter of returns
    axes.plot(sx5e_returns, [f_regression(x) for x in sx5e_returns ], 'r-') #plot the lin regression
    axes.set_xlabel('SX5E Returns') #change x label
    axes.set_ylabel('Div Fut Returns / Linear regression') #change y label
    axes.set_title('Linear Regression: Div Fut Returns = f(Spot returns) \n Sliding Maturity:'+str(sliding_maturity)+" years \n slope="+str(slope)) # change title
    fig.show() #show graph        
          
plot_linear_regression(8,df_STOXX50,df_futures_sliding) # EXample of plot using plot_linear_regression for sliding maturity= 8 years


#########################################################################################
############# Plot of the dividend slope term structure #################################
#########################################################################################
'''
In this section we compute the same dividend slope but we are repeating the process for all sliding maturities
between year 1 and year 8. Plotting the Slope as a function of the sliding maturity, we then observe that the Dividend
Slope is small on the short term but tends towards 1 on the long term.
'''
def plot_dividend_slope_termstructure(df_STOXX50, df_futures_sliding, list_sliding_maturity):
    list_div_slope=[]
    for sliding_maturity in list_sliding_maturity:
        slope=compute_linear_regression_spot_div(df_STOXX50, df_futures_sliding, sliding_maturity)[0]
        list_div_slope.append(slope)
        
    fig, axes = plt.subplots() #create plot object    
    axes.plot(list_sliding_maturity, list_div_slope, 'r*-') #plot the XY scatter of returns

    axes.set_xlabel('Maturity (years)') #change x label
    axes.set_ylabel('Dividend Slope') #change y label
    axes.set_title("Eurostoxx 50 Dividend Slope term structure") # change title
    fig.show() #show graph         
    

list_sliding_maturity=range(1,9)
plot_dividend_slope_termstructure(df_STOXX50, df_futures_sliding, list_sliding_maturity)
        