import matplotlib 
from smartbeta.backtest.group_factor_backtest import GroupFactorBackTest
import time
import pandas as pd
import datetime
import time
from smartbeta.analyst import ReturnAnalyzer

from data_provider.datafeed.universe import Universe
from data_provider.nestlib.trading_cal import TradeCal
from data_provider.datafeed.quote_feed import QuoteFeed
from data_provider.nestlib.market_info import Frequency
from smartbeta.factorbase import BaseFactor
from smartbeta.smartfactor import SmartFactor
from data_provider.nestlib.progress_bar import ProgressBar
import numpy as np
import pdb
from smartbeta.backtest.factor_backtest import FactorBackTest
import matplotlib.pyplot as plt
from smartbeta.construction.filters import *
import os 

class backtest_batch_service():
    # 生成因子回测结果并将结果以csv形式写入backtest_service_data文件夹中
    def run_backtest_batch(backtest_name,single_factor_name,start_date,end_date):
        r = os.popen('python bar_test.py '+backtest_name+' '+single_factor_name+' '+start_date+' '+end_date)
        r.read()
        

    
    # 将根据参数将结果dataframe读取出来，并plot图标，计算sharpe率
    def result_analysis(backtest_name,single_factor_name):
        df = pd.read_csv('backtest_service_data/'+backtest_name+'_'+single_factor_name+'.csv',index_col=[0])
        sharp_ratio = df['excess_ret'].mean()/df['excess_ret'].std()
        print('The sharp_ratio of '+single_factor_name+'= '+str(sharp_ratio))
        
        df = df.dropna()
        date_list = [datetime.datetime.strptime(str(elem), '%Y-%m-%d') for elem in df.index]
        
        df.excess_ret = (df.excess_ret + 1).cumprod()-1
        df.portfolio_ret = (df.portfolio_ret + 1).cumprod()-1
        df.benchmark_ret = (df.benchmark_ret + 1).cumprod()-1
        
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(211)
        ax.grid(True)
        ax.plot(date_list,df.excess_ret)
        ax.plot(date_list,df.portfolio_ret)
        ax.plot(date_list,df.benchmark_ret)
        plt.title(str(single_factor_name))
        plt.legend()
        plt.show()

    # list包含所有要分析回测的因子名字，plot_one如果为True，则将所有的得alpha曲线都plot在一个图里，方便比对，否则逐个plot并显示结果
    def batch_analysis(backtest_name,factor_name_li,plot_one):
        if plot_one == True:
            fig = plt.figure(figsize=(15,8))
            ax = fig.add_subplot(222)
            ax.grid(True)
            for factor in factor_name_li:
                df = pd.read_csv('backtest_service_data/'+backtest_name+'_'+factor+'.csv',index_col=[0])
                df = df.dropna()
                date_list = [datetime.datetime.strptime(str(elem), '%Y-%m-%d') for elem in df.index]
                df.excess_ret = (df.excess_ret + 1).cumprod()-1
                ax.plot(date_list,df.excess_ret,label=str(factor))
            plt.legend()
            plt.show()
        else:
            for factor in factor_name_li:
                fig = plt.figure(figsize=(15,8))
                ax = fig.add_subplot(222)
                ax.grid(True)
                df = pd.read_csv('backtest_service_data/'+backtest_name+'_'+factor+'.csv',index_col=[0])
                df = df.dropna()
                date_list = [datetime.datetime.strptime(str(elem), '%Y-%m-%d') for elem in df.index]
                df.excess_ret = (df.excess_ret + 1).cumprod()-1
                ax.plot(date_list,df.excess_ret)
                plt.title(str(factor))
                plt.show()


    



