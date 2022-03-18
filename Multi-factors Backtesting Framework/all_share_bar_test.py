
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
from __future__ import print_function, unicode_literals
import datetime
from data_provider.datafeed.universe import Universe
from dwtrader.core.engine import Engine



def change_time_format(x):
    changed_time = datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
    return changed_time


def initialize(context):
    """
    初始化
    """
    
    # 设置初始金额
    context.initial_capital = 10000000.
    
    # 订阅行情, 分别指定 bar周期 与 股票列表
    context.freq = '1D'
    # 订阅所有中证500成分股行情数据
    context.set_scope(all_share)
    
    # 设定参照基准
    context.benchmark = '000905.SH'
    # 设定回测起始时间
    context.start_date = sys.argv[3]
    context.end_date = sys.argv[4]
    
    # 设置交易费率
    context.set_cost(ratio=0.001)
    
    # 设定股票池刷新频率
    context.universe_freq = 'daily'
    

    
def build_universe(context):
    """
    择股逻辑
    """
    # # 先读取当前的中证500成分股
    # zz500 = context.get_components('000905.SH') 
    
    
    # 获取ep因子
    ep = context.factors[sys.argv[2]]
    
    # 从因子中选出排名前70的股票
    my_basket = ep.top(50, codes=all_share)
    context.universe.set(my_basket)
    

def handle_data(context, data):
    """
    策略逻辑
    """
    
    account = context.account
    # 需要同时遍历自选池中以及持仓中的股票
    codes = sorted(set(context.universe + context.holdings))
    
    for s in codes:
        if s in context.universe and account[s].position == 0:
            context.order_target_percent(s, 0.02)
            context.log('买入{}, 当前持仓{}手'.format(s, account[s].position))
         
        # 卖出在持仓中但不在股票池中的股票
        if s in context.holdings:
            if s not in context.universe: 
                context.order_target_value(s, 0)


date_time = datetime.datetime(2021, 2, 1)
uni = Universe()
all_share = uni.get_a_share_by_date(date_time)
# 默认会加载当前文件内的策略
engine = Engine()
engine.run()   

df = pd.DataFrame({'portfolio_ret':engine.get_portfolio_returns(),'benchmark_ret':engine.get_benchmark_returns()})
df.index = df.index.map(change_time_format)
df['excess_ret'] = df['portfolio_ret']-df['benchmark_ret']

df.to_csv('backtest_service_data/'+sys.argv[1]+'_'+sys.argv[2]+'.csv')