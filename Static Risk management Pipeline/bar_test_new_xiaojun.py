
"""
选择ep因子排名前50只股票，每月初调整股票池

10日均线上穿50日均线，买入持有；
10日均线跌破50日均线，卖出清仓。
"""

import numpy as np
import pandas as pd 
import sys
import time
from datetime import datetime

from collections import defaultdict, OrderedDict
from data_provider.datafeed.universe import Universe
from data_provider.nestlib.trading_cal import TradeCal
from smartbeta.smartfactor import SmartFactor

is_industry_neu = True 
is_amount_filter = True 

uni_handle=Universe()
tc_handle = TradeCal()

def change_time_format(x):
    changed_time = datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
    return changed_time

# 因为回测系统使用的是前复权的开盘价，这里计算市值必须也要用前复权的开盘价

def filter_by_amount(trading_day, capital=120000):
    """
    通过模拟撮合的量进行过滤，默认设置12万
    """
    try:
        df = SmartFactor('shared_simulated_matching_amount').load(trading_day,trading_day)
        ls = df[df['factor_value']>capital]['security_code'].tolist()
        return ls
    
    except:
        print(trading_day + ' 无法获取模拟撮合量')
        return []
    

def get_ticker_industry_map(trading_day):
    """股票代码和申万一级行业代码的映射,提供全A股的所有的股票映射
    """
    sw_df = uni_handle.get_sw_industry(trading_day)
    se = sw_df.set_index('securityId')['swIndustrycodeLv1']
    return se.to_dict()

def get_industry_max_buy(trading_day, capital=600000):
    result = {}
    sw_df = uni_handle.get_sw_industry(trading_day)
    for indu_code in sw_df['swIndustrycodeLv1'].unique():
        result[indu_code] = capital
        
    return result    


def filter_by_industry_constraint(context, factor_data):
    """通过行业进行约束,同时考虑股票池自身全部下单后超出限制
    """
    fa_df = factor_data.copy()
    industry_market_value = defaultdict(lambda: 0)
    
    today = context.now.strftime('%Y%m%d')
    tk2industry = get_ticker_industry_map(today)
    industry_max_buy = get_industry_max_buy(today)
    
    common = set(context.holdings).intersection(set(context.keep_list))
    for ticker in common:
        try:
            industry_market_value[tk2industry[ticker]] += context.account[ticker].position * context.account[ticker].open_price
        except Exception:
            continue
        
    fa_df = fa_df[~fa_df.index.isin(common)]  
    buy_num = context.stock_count-len(common)
    print('行业过滤前应该买入的股票数量,因子数量'+str(buy_num)+','+str(len(fa_df)))
    to_buy_ls = fa_df.index.tolist()
    
    
    ret_ls = []
    
    for tk in to_buy_ls:
        industry_code = tk2industry.get(tk)
        if not industry_code:
            print(tk + ' 没有对应的行业分类')
            continue
        
        if industry_market_value[industry_code] < industry_max_buy[industry_code]: 
            ret_ls.append(tk)
            if len(ret_ls) == buy_num:
                break
            industry_market_value[industry_code] += context.current_capital / context.stock_count
            
#     print(industry_market_value)
    print('行业过滤后的股票数量'+str(len(ret_ls)))

    return ret_ls


turn_over_ls = []
"""
选择shared_finance_xgb_6m因子排名前30只股票，每月初调整股票池
"""

import numpy as np
import pandas as pd


def initialize(context):
    """
    初始化
    """
    
    # 设置初始金额
    context.initial_capital = 10000000.
    
    # 订阅行情, 分别指定 bar周期 与 股票列表
    context.freq = '1D'
    # 订阅全A股行情数据
    context.set_scope("A")
    
    # 设定参照基准
    context.benchmark = "000905.SH"
    # 设定回测起始时间
    context.start_date = sys.argv[3]
    context.end_date = sys.argv[4]
    
    # 设置交易费率
    context.set_cost(ratio=0.0015)
    
    # 设定股票池刷新频率
    context.universe_freq = 'daily'
    
    # 历史持仓数据
    context.history_holdings = pd.DataFrame()

    context.is_order_day=True
    context.h_count=350
    context.keep_list_rank=600
    
    context.stock_count=100
    
def build_universe(context):
    """
    择股逻辑
    """
    factor = context.factors[sys.argv[2]]
    
    filtered=factor.get_value().sort_values(ascending=True)
    selected=filtered[:context.h_count]
    context.universe.set(selected.index.values.tolist())
    context.keep_list=filtered[:context.keep_list_rank].index.values.tolist()

    context.neu_adj = context.factors.get_multiple('shared_neu_geye_L1')
    context.orgin_adj = context.factors.get_multiple('shared_adjust_factor')
    
    # 设置当天为调仓日
    context.is_order_day=True
    
def handle_data(context, data):
    """
    策略逻辑
    """
    print(context.now)
    today = context.now.strftime('%Y%m%d')
    if not context.is_order_day:
        context.is_order_day=True
        print('returned')
        context.saved_capital=context.account.current_capital
        return
    account = context.account
    
    adj=context.orgin_adj.loc[context.now]
    sw_industry = uni_handle.get_sw_industry(context.now)
    
    sw_industry=sw_industry.set_index('securityId')
    industry_factor_data=pd.concat([adj,sw_industry], axis=1, join='inner')
    industry_mean_adj=industry_factor_data.groupby('swIndustryLv1')['shared_adjust_factor'].mean().sort_values()
    remove_industry = industry_mean_adj[:2].index.values
    
    today = context.now.strftime('%Y%m%d')
    
    try:
        factor_data=context.neu_adj.loc[context.now].reindex(context.universe).dropna().sort_values(by='shared_neu_geye_L1')
        

        if is_amount_filter:
            num1 = len(factor_data)
            print('成交量过滤前的数量'+str(num1))
            amount_filter_list = filter_by_amount(today)
            if len(amount_filter_list) > 1:
                factor_data = factor_data[factor_data.index.isin(amount_filter_list)]
            num2 = len(factor_data)
            print('成交量过滤后的数量'+str(num2))
            if num2 < num1:
                print("###################  成交量过滤掉的股票数量 #####################:   "+str(num1 -num2))
            
    except Exception as e:
        print('当天tick数据异常，使用原始的adj数据')
        factor_data=context.orgin_adj.loc[context.now].reindex(context.universe).dropna().sort_values(by='shared_adjust_factor')
    
    factor_data=pd.concat([factor_data,sw_industry], axis=1, join='inner')
    factor_data=factor_data[~factor_data.loc[:,'swIndustryLv1'].isin(remove_industry)]
    print(factor_data.shape)
    print(remove_industry)
    
    factor_data=factor_data.dropna()
    common = set(context.holdings).intersection(set(context.keep_list)) 

    if is_industry_neu:
        picked_universe = filter_by_industry_constraint(context, factor_data)
    else:
        factor_data = factor_data[~factor_data.index.isin(common)]
        factor_data=factor_data.head(context.stock_count-len(common))
        picked_universe = factor_data.index.tolist()
       
    count=len(picked_universe)
    print('共同的持仓数量'+ str(len(common)))
    print('换手率： ' + str(count/context.stock_count))
    print('当前持仓个数:'+str(len(context.holdings)))
    turn_over_ls.append(count/context.stock_count)
    
    to_sell = set(context.holdings) - common
    

    for si in to_sell:
        context.order_target_value(si, 0)
        if today ==context.end_date:
            print('卖出{}, 当前持仓{}手'.format(si, account[si].position))
    for si in picked_universe:
        if today ==context.end_date:
            print('买入前{}, 持仓{}手'.format(si, account[si].position))
        context.order_target_percent(si, 1/context.stock_count)
        if today ==context.end_date:
            print('买入后{}, 持仓{}手'.format(si, account[si].position))

def after_trading_end(context):
    """
    每个交易日收盘后触发
    """
    
    holding_assets = []
    
    acc = context.account
    td = context.now.strftime('%Y%m%d')

    for s in context.holdings:
        item = {}
        holding_assets.append(item)
        item['ticker'] = s
        item['amount'] = acc[s].position * acc[s].contract_unit
        item['close'] = acc[s].last_price
        item['datetime_str'] = td
        item['benchmark'] = context.benchmark
    
    # 持仓添加现金
    cash_item = {}
    holding_assets.append(cash_item)
    cash_item['ticker'] = 'CNY' 
    cash_item['amount'] = acc.cash
    cash_item['close'] = 1.
    cash_item['datetime_str'] = td
    cash_item['benchmark'] = context.benchmark

    holding_stats = pd.DataFrame(holding_assets)
    
    # 记录每日持仓
    context.history_holdings = context.history_holdings.append(holding_stats)

from dwtrader.core.engine import Engine
# 默认会加载当前文件内的策略
engine = Engine()
engine.run()   

df = pd.DataFrame({'portfolio_ret':engine.get_portfolio_returns(),'benchmark_ret':engine.get_benchmark_returns()})
df.index = df.index.map(change_time_format)
df['excess_ret'] = df['portfolio_ret']-df['benchmark_ret']

df.to_csv('backtest_service_data/'+sys.argv[1]+'_'+sys.argv[2]+'.csv')