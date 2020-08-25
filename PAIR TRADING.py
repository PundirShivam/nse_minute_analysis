"""
Analytics required for identifying a pair for pair - trading strategy.

@author : shivam pundir
@date   : 23-Aug-2020

"""

__verions__ = "0.0.1"

import pandas as pd
import numpy as np
import scipy.stats as stats
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
from statsmodels.tsa import stattools 



data = pd.read_csv(r"D:\strategy_rahul\PAIR TRADING\Nifty niftyBank 1 min 2015june to 2020  june.csv",index_col='date')
data.index = pd.to_datetime(data.index)
data = data.drop('Unnamed: 0',1)

def get_spread(price_one,price_two):
    '''
    function returns the spread from two 
    price series.
    
    Paramters
    -------------
    -------------
    price_one : pd.Series , pd.frame with single column
                i.e. close price series for asset one
    
    price_two : pd.Series , pd.frame with single column
                i.e. close price series for asset two
    
    Returns 
    ------------
    ------------
    spread  : pd.Series , spread from price series
    
    
    Defintion 
    ----------
    ----------
    spread: delta(price_one) - delta(price_two)
    
    '''
    spread = price_one.diff() - price_two.diff()
    return spread

def get_differential(price_one,price_two):
    '''
    function returns the differential from two 
    price series.
    
    Paramters
    -------------
    -------------
    price_one : pd.Series , pd.frame with single column
                i.e. close price series for asset one
    
    price_two : pd.Series , pd.frame with single column
                i.e. close price series for asset two
    
    Returns 
    ------------
    ------------
    differential  : pd.Series , spread from price series
    
    
    Defintion 
    ----------
    ----------
    differential: price_one - price_two
    
    '''
    differential =  price_one - price_two
    return differential

def get_ratio(price_one,price_two):
    '''
    function returns the ratio from two 
    price series.
    
    Paramters
    -------------
    -------------
    price_one : pd.Series , pd.frame with single column
                i.e. close price series for asset one
    
    price_two : pd.Series , pd.frame with single column
                i.e. close price series for asset two
    
    Returns 
    ------------
    ------------
    ratio  : pd.Series , spread from price series
    
    
    Defintion 
    ----------
    ----------
    ratio: price_one/price_two
    
    '''
    ratio =  price_one*1./price_two
    return ratio

def get_correlation(price_one,price_two,return_based_corr=True,method='pearson'):
    '''
    function returns the correlation from two 
    price series.
    
    Paramters
    -------------
    -------------
    price_one : pd.Series , pd.frame with single column
                i.e. close price series for asset one
    
    price_two : pd.Series , pd.frame with single column
                i.e. close price series for asset two
    
    return_based_corr : boolean, if True returns a dictionary with
                correlation based on price series & return series .
                if False, function returns a scalar value which is
                correlation of passed series.
                
    method : string, "pearson" or "spearman" correaltion of
             price series
    
    Returns 
    ------------
    ------------
    correlation  : float (scalar) or tuple  , correlation between
                   two price/return series
        
    '''
    correlation  = price_one.corr(price_two,method=method)
        
    if return_based_corr:
        return_series_one = price_one.pct_change()
        return_series_two = price_two.pct_change()
        return_based_correaltion = return_series_one.corr(return_series_two,method=method)
        correlation = (correlation,return_based_correaltion)

    return correlation

def get_density_curve(recent_value,mean,standard_deviation):
    '''
    The function returns the density curve 
    value (scalar).
    
    Parameters
    -----------
    -----------
    recent_value : float , scalar or vector
    mean         : float , scalar or vector
    standard_deviation : float, scalar or vector
    
    Returns
    -----------
    -----------
    density_curve : float , scalar or vector
    
    Definition
    -----------
    -----------
    Density curve is the cdf for the recent
    value with a mean and standard_deviation
    assuming normal distribution.
    
    '''
    
    mydist = stats.norm(mean,standard_deviation )
    density_curve = mydist.cdf(recent_value)
#     density_curve = norm.cdf(recent_value,loc=mean,
#                             scale=standard_deviation)
    return density_curve

def get_pair_data(price_one,price_two):
    
    '''
    The function returns the pair data from  price 
    series of two individual stocks. Please ensure 
    that price series is adjusted for corporate actions 
    like splits, dividends etc. As those adjustments 
    are not included in this function. And, using unadjusted 
    prices can lead to erronous conclusions.

    Paramters
    -------------
    -------------
    price_one : pd.Series , pd.frame with single column
                i.e. close price series for asset one
    
    price_two : pd.Series , pd.frame with single column
                i.e. close price series for asset two
    
    '''
    # Generating Pair Tracking series 
    # Spread, Differential , Ratio
    functions = [get_spread,get_differential,get_ratio]
    function_names = ['Spread','Differential','Ratio']
    function_series_list = []
    for name,func in zip(function_names,functions):
        function_series_list.append(func(price_one,price_two))
    pair_tracking = pd.concat(function_series_list,axis=1).dropna()
    pair_tracking.columns = function_names
    
    # correlation
    correlation = get_correlation(price_one,price_two,return_based_corr=True,method='pearson')
    correlation = pd.DataFrame(correlation,index=['Close','% Return'],columns=['Correaltion'])
    
    # Basic Stats
    mean = pair_tracking.mean()
    median = pair_tracking.median()
    standard_deviation = pair_tracking.std()
    absolute_deviation = (pair_tracking-mean).abs().mean()
    
    basic_stats = pd.concat([mean,median,standard_deviation,
                            absolute_deviation],axis=1)
    basic_stats.columns = ['Mean','Median','Standard Deviation','Absolute Deviation']
    
    # standard deviations
    standarad_deviations_range =  np.arange(-3,4)
    standard_deviation_columns = ['{:+}'.format(std) if std else 'Mean' for std in standarad_deviations_range ] 
    standard_deviation_list  = []
    for std in standarad_deviations_range:
        standard_deviation_list.append(mean + std*standard_deviation)
    standard_deviation_table  = pd.concat(standard_deviation_list,axis=1)    
    standard_deviation_table.columns  = standard_deviation_columns
    standard_deviation_table  =  standard_deviation_table.transpose()
    
    return pair_tracking,correlation,basic_stats,standard_deviation_table


from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

def linear_regression(y,x,intercept=True):
    '''
    the function returns the 
    fit object from statsmodel api    
    
    Paramters
    ----------------
    ----------------
    y : pd.Series object,
        dependent variable
        
    x : pd.Series or pd.Dataframe object,
        independent variable
    
    intercept: boolean, True / False
        if True, an intercept term will be 
        added as "const" in independent 
        variables
        
    Return 
    ----------------
    ----------------
    fit : statsmodels.regression.linear_model.RegressionResultsWrapper
    
    '''
    if intercept:
        x = sm.add_constant(x)
    fit = sm.OLS(y,x,missing='drop').fit()
    return fit


# In[14]:


# Error Ratio = Standard Error of Intercept / Standard Error
def get_error_ratio(price_one,price_two,get_fit=False):
    '''
    Function returns the error ratio
    for a pair of price series.
    
    Paramters
    -------------
    -------------
    price_one : pd.Series , pd.frame with single column
                i.e. close price series for asset one
                It will be the Y in the regression 
                i.e. dependent variable
    
    price_two : pd.Series , pd.frame with single column
                i.e. close price series for asset two
                It will be the X in the regression 
                i.e. independent variable
    
    get_fit : boolean, True/False
                if True, stats model fit object
                is also returned
    
    Returns
    ----------------
    ----------------
    error_ratio : scalar
    
    fit : statsmodels.regression.linear_model.RegressionResultsWrapper 
          (optional), returned if get_fit is True
    
    Definition
    ------------------
    ------------------
    Error Ratio = Standard Error of Intercept /
                        Standard Error of the residuals
                        
    Regression
    ------------------
    ------------------
    
    price_two = slope * price_one + intercept 
    
    '''
    
    fit = linear_regression(price_one,price_two,intercept=True)
    standard_error_intercept = fit.bse['const']
    standard_error_resid =  np.std(fit.resid)
    error_ratio = standard_error_intercept/standard_error_resid
    
    if get_fit:
        return error_ratio, fit 
    return error_ratio

def identify_x_and_y(price_one,price_two,get_fit=False):
    '''
    The function returns True if price_one is
    x else False if price_one if True
    
    Paramters
    -------------
    -------------
    price_one : pd.Series , pd.frame with single column
                i.e. close price series for asset one
    
    price_two : pd.Series , pd.frame with single column
                i.e. close price series for asset two
    
    get_fit : boolean, True/False
                if True, stats model fit object
                is also returned
    
    Returns
    ----------------
    ----------------
    price_one_x_flag : boolean, True/False 
    
    fit : statsmodels.regression.linear_model.RegressionResultsWrapper 
          (optional), returned if get_fit is True
    
    Note
    ------------------
    ------------------
    price_two = slope * price_one + intercept 
    
    Choice of x,y in regression is based on 
    combination with results in lowest error
    ratio.
    
    See: get_error_ratio for more detail.
    '''
    price_one_as_x_fit,fit_one  = get_error_ratio(price_two,price_one,get_fit=True)
    price_two_as_x_fit,fit_two  = get_error_ratio(price_one,price_two,get_fit=True)
    price_one_x_flag = price_one_as_x_fit < price_two_as_x_fit
    if get_fit:
        return price_one_x_flag , (fit_one if price_one_x_flag else fit_two)    
    return price_one_x_flag


# In[15]:


'''
There are two conditions for stationarity.

1. The mean of the series should be same or within a tight range.
2. The standard deviation should be within a range.
3. There should be no autocorrelation within the series.

'''


# In[16]:


def get_hurst_exponent(ts):
    """
    Returns the Hurst Exponent of the time series vector ts
    
    Parameters
    -------------
    -------------
    ts : pd.Series , numpy array 
    
    Returns
    -------------
    -------------
    hurst_exponent: scalar (float)
    
    Note
    -------------
    --------------
    A time series can charachterised in the following manner:
    * H < 0.5 - The time seris in mean reverting
    * H = 0.5 - The time series is a Geometric Browninan Motion
    * H > 0.5 - The time series is trending
    
    H near 0 is a highly mean reverting series, while for H
    near 1 the series is trongly trending.
    
    Reference
    -------------
    -------------
    weblink 
    https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    hurst_exponent =  poly[0]*2.0
    return hurst_exponent

def get_stationarity_tests(ts):
    '''
    Returns the Hurst Exponent and ADF (pvalue) of the time series vector ts
    Parameters
    -----------
    -----------
    ts : pd.Series , np.array
    Returns 
    -----------
    -----------
    tests : tupple i.e (Hurst_exponent,ADF)

    '''
    hurst_exponent = get_hurst_exponent(ts)
    adf = stattools.adfuller(ts)[1]
    ## TODO: Addition of tests Johsen Co-integration test
    
    return hurst_exponent,adf

identify_x_and_y(nifty,bank_nifty)
get_error_ratio(bank_nifty,nifty)
fit  = linear_regression(nifty,bank_nifty);fit.bse['const']/np.std(fit.resid)
from statsmodels.tsa.vector_ar.vecm import coint_johansen
get_stationarity_tests(nifty+bank_nifty)
nifty = data['nifty_close']
bank_nifty = data['NiftyBANK_close']
nifty = nifty.ffill().resample('B').last().dropna()
bank_nifty = bank_nifty.ffill().resample('B').last().dropna()
pair_tracking , correlation,basic_stats,standard_deviation_table = get_pair_data(nifty,bank_nifty)
correlation
basic_stats
standard_deviation_table
ratio  = pair_tracking[['Ratio']]
ratio['mean'] = ratio['Ratio'].rolling(250).mean().shift()
ratio['std'] = ratio['Ratio'].rolling(250).std().shift()
ratio = ratio.dropna()
ratio['density_curve'] = get_density_curve(ratio['Ratio'],ratio['mean'],ratio['std'])
ratio['density_curve'].plot(figsize=(12,8),grid=True)



