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

class backtest_batch_service():
    def run_backtest_batch(backtest_name,single_factor_name,start_date,end_date):
        #接收前端传入参数
        start = start_date
        end = end_date
        freqValue = 'daily'
        benchmark= '000905.SH'
        tkScope = 'A'
        st = False
        limitDown = False
        neutral = False
        holding_count = 200
        tradeCost = 0.0001
        industry = 'all'
        range_factors = [{'name': single_factor_name, 'direction': -1}]
        filter_factors = []
        filter_factors_info = []
        bt_name = backtest_name
        timeperiod = False

        starttemp = time.strptime(str(start), '%Y-%m-%d')     
        startDate = time.strftime('%Y%m%d', starttemp) 
        endtemp = time.strptime(str(end), '%Y-%m-%d')     
        endDate = time.strftime('%Y%m%d', endtemp) 

        freq = freqValue
        tk_scope = tkScope
        industry_value = industry

        if freq == 'daily':
            freq ='每日调仓'
        elif freq == 'week_end':
            freq = '每周调仓'
        else:
            freq = '每月调仓'

        if tk_scope == 'A':
            tk_scope = '全A股'

        if industry_value == 'all':
            industry_value = '全行业'


        uni = Universe()
        start_dt=pd.to_datetime(startDate)
        end_dt=pd.to_datetime(endDate)
        uni = Universe()
        all_A_shares = uni.get_a_share_in_period(start_dt,end_dt)

        bt = FactorBackTest(benchmark,startDate,endDate,range_factors,holding_count=holding_count,freq=freqValue)

        if timeperiod:
            ns = NewStockFilter()
            ns.set_ipo_date_timeperiod(timeperiod)
            bt.add_filter_rules(ns)

        # 设置因子过滤条件
        if len(filter_factors)>0:
            ff= FactorFilter(filter_factors)
            ff.set_interval(startDate, endDate)
            vf = ValueFilter()
            pf = PercentileFilter()

            for ft in filter_factors_info:
                factor_name = ft['name']
                if 'factor_parameters' in ft and len(ft['factor_parameters'])>0:
                    factor_name = ft['name'] + str('_') + str(ft['factor_parameters'])
                if 'percentilefilter' in ft and len(ft['percentilefilter'])>0:
                    pf.set_interval(ft['percentilefilter']['valuefrom'], ft['percentilefilter']['valueto'])
                    ff.add_filter_for_spec_factor(factor_name, pf)
                if 'valuefilter' in ft and len(ft['valuefilter'])>0:
                    vf.set_interval(ft['valuefilter']['valuefrom'], ft['valuefilter']['valueto'])
                    ff.add_filter_for_spec_factor(factor_name, vf)

            bt.add_filter_rules(ff)

        #目前买卖限制
        bt.set_buy_limit_down(limitDown)#如果为True
        bt.set_buy_st(st)  # 不买被st的股票
        bt.set_select_scope(tkScope)  # 从全A股中选取
        bt.set_commission(buy_cost=tradeCost, sell_cost=tradeCost)
        bt.set_industry_neutralization(neutral)
        if not industry == 'all':
            bt.set_industry_v1_code(industry)

        bt.run()
        
        

        rets = pd.DataFrame()
        rets['datetime'] = list(bt.portfolio_ret.keys())
        rets['month'] = rets.datetime.apply(lambda x:x.year*100+x.month)
        rets['portfolio_ret'] = bt.portfolio_ret.values
        rets['benchmark_ret'] = bt.benchmark_ret.values
        rets['excess_ret'] = rets.portfolio_ret - rets.benchmark_ret
        rets.index = rets.datetime

        rets.to_csv('backtest_service_data/'+backtest_name+'_'+single_factor_name+'.csv')
    
    def result_analysis(backtest_name,single_factor_name):
        df = pd.read_csv('backtest_service_data/'+backtest_name+'_'+single_factor_name+'.csv')
        sharp_ratio = df['excess_ret'].mean()/df['excess_ret'].std()
        print('The sharp_ratio of '+single_factor_name+'= '+str(sharp_ratio))
        
        df = df.dropna()
        date_list = [datetime.datetime.strptime(str(elem), '%Y-%m-%d') for elem in df.datetime]
        
        df.excess_ret = (df.excess_ret + 1).cumprod()-1
        df.portfolio_ret = (df.portfolio_ret + 1).cumprod()-1
        df.benchmark_ret = (df.benchmark_ret + 1).cumprod()-1
        # fig = plt.figure(figsize=(15,8))
        # ax = fig.add_subplot(211)
        # ax.grid(True)
        # ax.plot(date_list,df.excess_ret)
        # ax.plot(date_list,df.portfolio_ret)
        # ax.plot(date_list,df.benchmark_ret)
        # plt.title(str(single_factor_name))
        # plt.legend()
        # plt.show()
        return [single_factor_name,sharp_ratio,df.excess_ret[-1]]


    def batch_analysis(backtest_name,factor_name_li,plot_one):
        if plot_one == True:
            fig = plt.figure(figsize=(15,8))
            ax = fig.add_subplot(222)
            ax.grid(True)
            for factor in factor_name_li:
                df = pd.read_csv('backtest_service_data/'+backtest_name+'_'+factor+'.csv')
                df = df.dropna()
                date_list = [datetime.datetime.strptime(str(elem), '%Y-%m-%d') for elem in df.datetime]
                df.excess_ret = (df.excess_ret + 1).cumprod()-1
                ax.plot(date_list,df.excess_ret,label=str(factor))
            plt.legend()
            plt.show()
        else:
            for factor in factor_name_li:
                fig = plt.figure(figsize=(15,8))
                ax = fig.add_subplot(222)
                ax.grid(True)
                df = pd.read_csv('backtest_service_data/'+backtest_name+'_'+factor+'.csv')
                df = df.dropna()
                date_list = [datetime.datetime.strptime(str(elem), '%Y-%m-%d') for elem in df.datetime]
                df.excess_ret = (df.excess_ret + 1).cumprod()-1
                ax.plot(date_list,df.excess_ret)
                plt.title(str(factor))
                plt.show()


    



