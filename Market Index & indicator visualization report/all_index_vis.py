import pandas as pd
import datetime
from tqdm import tqdm
import csv
from pandas import Series
from datetime import datetime
import random
import time 
import statsmodels.api as sm
import pdb
from data_provider.datafeed.quote_feed import QuoteFeed
from data_provider.nestlib.trading_cal import TradeCal
from data_provider.datafeed.universe import Universe
from data_provider.nestlib.market_info import Frequency
from data_provider.datafeed.financial_feed import FinancialFeed
from data_provider.datafeed.finance_utils import ttmContinues, ttmDiscrete
from smartbeta.smartfactor import SmartFactor
from index_li import mao_40_li
from index_li import institutions_baotuan_50_li
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec#分割子图
from data_provider.nestlib.oracle_util import OracleConnection, transfer_column_lower
from datetime import date, timedelta
import os
import seaborn as sns
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
uni = Universe()
tc = TradeCal()

def get_new_ipo_stocks(date):
    ipo_df = uni.get_all_ipo_info()
    limit = tc.shift_date(date, 30, direction='backward')
    new_ipo_stocks_df = ipo_df[ipo_df['ipo_date']>int(limit)]
    new_ipo_stocks_li = new_ipo_stocks_df['ticker'].tolist()
    return new_ipo_stocks_li


def get_number_of_all_stocks(date):
#     print('正在获取每日股票总数(除去新股)。。。。。。')
    all_stocks_li = uni.get_a_share_by_date(date)
    all_stocks_li = list(set(all_stocks_li)-set(get_new_ipo_stocks(date)))
    return all_stocks_li


def get_number_of_up_stocks_all_stcoks(begin_date,end_date):
    print('正在获取每日上涨的股票数和每日股票总数(除去新股)。。。。。。')
    df = SmartFactor('dailyreturn').load(begin_date,end_date)
    gb_df_data = df.groupby('tdate')
    date_number_li = []
    for any_date,date_df in tqdm(gb_df_data):
        date_df = date_df.copy()
        date_df = date_df[date_df['factor_value']>0]
        date_df = date_df.drop(date_df[date_df['security_code'].isin(get_new_ipo_stocks(any_date))].index)
        number = len(date_df)
        number_all = len(get_number_of_all_stocks(any_date))
        date_number_li.append([any_date,number,number_all])
    result = pd.DataFrame(date_number_li,columns=['date','number_of_up_stocks','number_of_all_stocks'])
    return result


def get_index_dailyreturn(ticker,begin_date,end_date):    
    tickers = ticker
    shift_back_begin_date = tc.shift_date(begin_date, 1, direction='backward')
    week_quote = QuoteFeed(
        universe_ticker=tickers,
        begin_day=shift_back_begin_date,
        end_day=end_date,
        tracking_freq=86400,
        is_index=True,
    )
    week_quote.load_feed()
    df = week_quote.get_index_quote()
    df.loc[:,'yesterday_close'] = df['close'].shift(1)
    df = df.dropna()
    df.loc[:,'dailyreturn'] = (df['close']-df['yesterday_close'])/df['yesterday_close']
    return df[['datetime_str','dailyreturn']]


def get_median_dailyrertun_of_all_market(begin_date,end_date):
    all_stocks_df = SmartFactor('dailyreturn').load(begin_date,end_date)
    gb_df_data = all_stocks_df.groupby('tdate')
    zz500_df = get_index_dailyreturn('000905.SH',begin_date,end_date)
    hs300_df = get_index_dailyreturn('000300.SH',begin_date,end_date)
    date_return_li = []
    for any_date,date_df in tqdm(gb_df_data):
        date_df = date_df.copy()
        median_return = date_df['factor_value'].median()
        hs_300_return = hs300_df.loc[hs300_df[hs300_df['datetime_str']==any_date].index,'dailyreturn'].values[0]
        zz_500_rertun = zz500_df.loc[zz500_df[zz500_df['datetime_str']==any_date].index,'dailyreturn'].values[0]
        date_return_li.append([any_date,median_return,hs_300_return,zz_500_rertun])
    result = pd.DataFrame(date_return_li,columns=['date','all_market_median_return','hs_300_return','zz_500_return'])
    return result    


def get_special_index_return(ticker_li,begin_date,end_date):
    all_stocks_df = SmartFactor('dailyreturn').load(begin_date,end_date)
    special_index_df = all_stocks_df.loc[all_stocks_df[all_stocks_df['security_code'].isin(ticker_li)].index,:]
    data = special_index_df.groupby('tdate')
    return_li = []
    for any_date,date_df in data:
        mean_return = date_df['factor_value'].mean()
        return_li.append([any_date,mean_return])
    return_df = pd.DataFrame(return_li,columns=['date','special_index_mean_dailyreturn'])
    return return_df


def get_all_market_amount(begin_date,end_date):
    all_tickers = uni.get_a_share_in_period(begin_date, end_date)
    quote = QuoteFeed(
    universe_ticker=all_tickers,
    begin_day=begin_date,
    end_day=end_date,
    tracking_freq=86400,
    use_cache=True,
    )
    quote.load_feed()
    all_df = quote.get_stock_quote()
    data = all_df.groupby('datetime')
    amount_li = []
    for any_date,date_df in data:
        total_amount = date_df['amount'].sum()
        mao_40_amount =date_df.loc[date_df[date_df['ticker'].isin(mao_40_li)].index,'amount'].sum()
        institutions_baotuan_50_amount =date_df.loc[date_df[date_df['ticker'].isin(institutions_baotuan_50_li)].index,'amount'].sum()
        amount_li.append([any_date,mao_40_amount,institutions_baotuan_50_amount,total_amount])
    result = pd.DataFrame(amount_li,columns=['date','mao_40_amount','institutions_baotuan_50_amount','all_stocks_total_daily_amount'])
    result.loc[:,'mao_40_ratio'] = result['mao_40_amount']/result['all_stocks_total_daily_amount']
    result.loc[:,'institutions_baotuan_50_ratio'] = result['institutions_baotuan_50_amount']/result['all_stocks_total_daily_amount']
    return result


def get_fin_sec_info(begin_date,end_date):
    tradeday_li = tc.get_trading_day_list(begin_date,end_date)
    df = pd.read_csv('fin_sec_info.csv')
    result = df.loc[df[df['date'].isin(tradeday_li)].index,:]
    result = result.sort_values(by='date',ascending=False)
    return result[['date','finval_sum','secval_sum','finval_secval_sum','fin_sec_ratio']] #日期,融资余额,融券余额,融资融券余额,融资占融资融券余额比例


def get_number_ratio_up_all_stocks(df):
    pic_1_df = df[['date','number_of_up_stocks','number_of_all_stocks']]
    pic_1_df = pic_1_df.reset_index(drop=True)
    pic_1_df = pic_1_df.set_index('date').sort_index(ascending=True)
    pic_1_df['number_of_down_stocks'] = -(pic_1_df['number_of_up_stocks'] - pic_1_df['number_of_all_stocks'])
    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(111)
    bar1 = ax1.bar(np.arange(len(pic_1_df.index)),pic_1_df['number_of_up_stocks'],alpha=0.5, width=0.3, color='yellow', edgecolor='red', label='上涨股票数量', lw=5)
    bar2 = ax1.bar(np.arange(len(pic_1_df.index))+0.5,pic_1_df['number_of_down_stocks'], alpha=0.2, width=0.3, color='g', edgecolor='green', label='下跌股票数量', lw=5)
    ax1.set_xticks(range(0,len(pic_1_df.index),1))
    ax1.set_xticklabels([pic_1_df.index.tolist()[index] for index in ax1.get_xticks()])#标签设置为日期
    for rect in bar1 + bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom',fontsize=10)

    ax1.legend()
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(10)#设置标签字体
    ax1.set_facecolor('white')
    plt.show()


def get_zz500_hs300_all_market_compare(df):
    pic_2_df = df[['date','zz500_dailyreturn','hs300_dailyreturn','all_market_median_return','all_stocks_total_daily_amount']]
    pic_2_df = pic_2_df.reset_index(drop=True)
    pic_2_df = pic_2_df.iloc[:200,:]
    pic_2_df['zz500_dailyreturn'] = pic_2_df['zz500_dailyreturn'] + 1
    pic_2_df['hs300_dailyreturn'] = pic_2_df['hs300_dailyreturn'] + 1
    pic_2_df['all_market_median_return'] = pic_2_df['all_market_median_return'] + 1
    pic_2_df['date'] = pic_2_df['date'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d'))
    pic_2_df['date'] = pd.to_datetime(pic_2_df['date'])

    pic_2_df = pic_2_df.rename(columns={'date':'日期','zz500_dailyreturn':'中证500收益','hs300_dailyreturn':'沪深300收益','all_market_median_return':'全市场中位数收益','all_stocks_total_daily_amount':'全市场成交额'})

    pic_2_df = pic_2_df.set_index('日期')
    fig = plt.figure()
    ax1=fig.add_subplot()
    pic_2_df = pic_2_df.sort_index(ascending=True)
    ax1 = pic_2_df[['中证500收益','沪深300收益','全市场中位数收益']].cumprod().plot(x=pic_2_df.index.astype(str),colormap='Accent',figsize=(16,8),grid=False,lw=5,fontsize=10)
    ax1.set_facecolor('white')

    ax2 = ax1.twinx()
    ax2.bar(np.arange(0, len(pic_2_df.index)), pic_2_df['全市场成交额'],alpha=0.5, color='red',label='全市场成交额')
    ax2.set_ylim(0,3000000000000)
    # ax2.set_xticks(range(0,len(pic_2_df.index),15))#X轴刻度设定 每15天标一个日期
    # ax2.set_xticklabels([pic_2_df.index.tolist()[index] for index in ax2.get_xticks()])#标签设置为日期
    plt.legend(loc='upper center')
    plt.grid(False)
    plt.show()
    # pic_2_df = df[['date','zz500_dailyreturn','hs300_dailyreturn','all_market_median_return']]
    # pic_2_df = pic_2_df.reset_index(drop=True)
    # pic_2_df = pic_2_df.iloc[:200,:]
    # pic_2_df['zz500_dailyreturn'] = pic_2_df['zz500_dailyreturn'] + 1
    # pic_2_df['hs300_dailyreturn'] = pic_2_df['hs300_dailyreturn'] + 1
    # pic_2_df['all_market_median_return'] = pic_2_df['all_market_median_return'] + 1
    # pic_2_df['date'] = pic_2_df['date'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d'))
    # pic_2_df['date'] = pd.to_datetime(pic_2_df['date'])
    # pic_2_df = pic_2_df.rename(columns={'date':'日期','zz500_dailyreturn':'中证500收益','hs300_dailyreturn':'沪深300收益','all_market_median_return':'全市场中位数收益'})
    # pic_2_df = pic_2_df.set_index('日期').sort_index(ascending=True)

    # ax = pic_2_df.cumprod().plot(x=pic_2_df.index.astype(str),colormap='Accent',grid=True,figsize=(16,6),lw=5,fontsize=10)
    # ax.set_facecolor('white')


def get_baotuan50_mao40_ret_plot(df):
    df_stockload = df
    # df_stockload = df_stockload.iloc[:100,:]

    fig = plt.figure(figsize=(16,6), dpi=100)#创建fig对象

    gs = gridspec.GridSpec(5, 1, left=0.06, bottom=0.15, right=0.96, top=0.96, wspace=None, hspace=0)
    graph_Line = fig.add_subplot(gs[:3,:],facecolor='white')
    graph_Vol_baotuan = fig.add_subplot(gs[3,:],facecolor='white')
    graph_Vol_mao = fig.add_subplot(gs[4,:],facecolor='white')

    df_stockload = df_stockload.set_index('date')
    df_stockload = df_stockload.sort_index(ascending=True)

    df_stockload['institutions_baotuan_50_mean_dailyreturn'] = df_stockload['institutions_baotuan_50_mean_dailyreturn'] + 1
    df_stockload['mao_40_mean_dailyreturn'] = df_stockload['mao_40_mean_dailyreturn'] + 1
    df_stockload['institutions_baotuan_50_mean_dailyreturn'] = df_stockload['institutions_baotuan_50_mean_dailyreturn'].cumprod()
    df_stockload['mao_40_mean_dailyreturn'] = df_stockload['mao_40_mean_dailyreturn'].cumprod()


    graph_Line.plot(np.arange(0, len(df_stockload.index)),df_stockload['institutions_baotuan_50_mean_dailyreturn'],'blue', label='抱团_50',lw=2.0)
    graph_Line.plot(np.arange(0, len(df_stockload.index)),df_stockload['mao_40_mean_dailyreturn'],'green', label='茅_40',lw=2.0)

    graph_Line.legend(loc='best')
    # graph_Line.set_title("Baotuan50_Mao40_return_plot")
    graph_Line.set_ylabel("累积收益")
    graph_Line.set_xlim(0, len(df_stockload.index))
    graph_Line.set_xticks(range(0, len(df_stockload.index), 30))

    graph_ratio_line = graph_Line.twinx()
    graph_ratio_line.plot(np.arange(0, len(df_stockload.index)),df_stockload['mao_40_ratio'],'purple', label='茅股全市场成交额占比',lw=1.0)
    graph_ratio_line.plot(np.arange(0, len(df_stockload.index)),df_stockload['institutions_baotuan_50_ratio'],'pink', label='抱团股全市场成交额占比',lw=1.0)
    graph_ratio_line.legend(loc='center left')
    graph_ratio_line.set_xlim(0, len(df_stockload.index))
    graph_ratio_line.set_ylim(0,0.5) 

    graph_Vol_baotuan.bar(np.arange(0, len(df_stockload.index)), df_stockload.institutions_baotuan_50_amount, color='red')
    graph_Vol_baotuan.set_ylabel("抱团股成交额")
    # graph_Vol_baotuan.set_xlabel("date")
    graph_Vol_baotuan.set_xlim(0,len(df_stockload.index)) #设置一下x轴的范围

    graph_Vol_mao.bar(np.arange(0, len(df_stockload.index)), df_stockload.mao_40_amount, color='orange')
    graph_Vol_mao.set_ylabel("茅股成交额")
    # graph_Vol_mao.set_xlabel("date")
    graph_Vol_mao.set_xlim(0,len(df_stockload.index)) #设置一下x轴的范围
    graph_Vol_mao.set_xticks(range(0,len(df_stockload.index),30))
    graph_Vol_mao.set_xticklabels([df_stockload.index.tolist()[index] for index in graph_Vol_mao.get_xticks()])#标签设置为日期

    #X-轴每个ticker标签都向右倾斜45度
    for label in graph_Line.xaxis.get_ticklabels():
        label.set_visible(False)#隐藏标注 避免重叠                

    for label in graph_Vol_mao.xaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(10)#设置标签字体
    plt.grid(False)
    plt.show()



def get_market_indexs_plot(time_window):
    today = date.today()
    begin_day = today - timedelta(days=time_window)
    begin_day = begin_day.strftime('%Y%m%d')
    tickers = ['399372.SZ','399373.SZ','399375.SZ','399374.SZ','399376.SZ','399377.SZ']
    # time = ['20200106', '20210402']

    key_words = """s_info_windcode ticker,
                 trade_dt datetime,
                 s_dq_close close,
                 s_dq_pctchange return"""
    table_name = 'wind.AIndexEODPrices'
    # condition = "s_info_windcode in ('%s') and trade_dt >= ('%s')" % ("', '".join(tickers), "', '".join(time)) 
    condition = "s_info_windcode in ('%s') and trade_dt >= ('%s')" % ("', '".join(tickers),begin_day) 

    query = "SELECT %s FROM %s WHERE %s" % (key_words, table_name, condition)
    with OracleConnection() as conn: #注意：请务必使用此方式安全断开数据库连接，避免连结池耗尽。
        ret = pd.read_sql(query, con=conn)
        ret = transfer_column_lower(ret)
    ret = ret.rename(columns={'datetime':'日期'})
    ret = ret.set_index('日期').sort_index(ascending=False)
    
    index_map = {'399372.SZ':'大盘成长','399373.SZ':'大盘价值','399375.SZ':'中盘价值','399374.SZ':'中盘成长','399376.SZ':'小盘成长','399377.SZ':'小盘价值'}
    # index_map = {'399372.SZ':'big_cap_growth','399373.SZ':'big_cap_value','399375.SZ':'mid_cap_value','399374.SZ':'mid_cap_growth','399376.SZ':'small_cap_growth','399377.SZ':'small_cap_value'}

    gb_ticker_data = ret.groupby('ticker')
    combined_df = pd.DataFrame()
    for any_ticker,ticker_df in gb_ticker_data:
        ticker_df = ticker_df.copy()
        ticker_df = ticker_df.sort_index(ascending=True)
        ticker_df.loc[:,'return'] = (ticker_df['return']+100)/100
        ticker_df.loc[:,'cumprod_ret'] = ticker_df['return'].cumprod()
        combined_df.loc[:,index_map[any_ticker]+'累积收益'] = ticker_df['cumprod_ret']
    ax = combined_df.plot(figsize=(16,6))
    ax.set_facecolor('white')


def get_fin_sec_hs300_compare(df):
    pic_3_df = df[['date','finval_sum','secval_sum','finval_secval_sum','hs_300_return']]
    fig = plt.figure(figsize=(16,6))
    graph_Bar = fig.add_subplot(111)
    pic_3_df = pic_3_df.reset_index(drop=True)
    pic_3_df = pic_3_df.set_index('date')
    pic_3_df = pic_3_df.sort_index(ascending=True)
    pic_3_df = pic_3_df.iloc[:,:]

    graph_Bar.bar(np.arange(len(pic_3_df.index)),pic_3_df['finval_secval_sum'],alpha=0.5,width=0.7,color='red',label='融资融券余额')
    graph_Bar.bar(np.arange(len(pic_3_df.index)),pic_3_df['secval_sum'],alpha=0.5, width=0.7,color='blue',label='融券余额')
    graph_Bar.set_xticks(range(0,len(pic_3_df.index),30))
    graph_Bar.set_xticklabels([pic_3_df.index.tolist()[index] for index in graph_Bar.get_xticks()])#标签设置为日期
    graph_Bar.set_facecolor('white')
    graph_Bar.legend(loc='upper left')

    pic_3_df['hs_300_return'] = pic_3_df['hs_300_return']+1
    graph_Line =graph_Bar.twinx()
    graph_Line.plot(np.arange(0, len(pic_3_df.index)),pic_3_df['hs_300_return'].cumprod(),color='g',lw=2.0,label='沪深300收益曲线')
    graph_Line.legend(loc='best')
    graph_Line.set_xlabel('date')
    graph_Line.set_ylim(0,3)

    for label in graph_Bar.xaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(10)#设置标签字体
    plt.legend()
    plt.grid(False)
    plt.show()

def get_amount_bias_heatmap():
    df = pd.read_pickle('industry_amount_bias.pkl')
    fig, ax = plt.subplots(figsize = (16,6),dpi=200)  
    sns.heatmap(df, annot=True, ax=ax)


def get_number_limit_up_down(df):
    pic_4_df = df[['date','zhangting_number','dieting_number']]
    pic_4_df = pic_4_df.reset_index(drop=True)
    pic_4_df = pic_4_df.set_index('date').sort_index(ascending=True)
    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(111)
    bar1 = ax1.bar(np.arange(len(pic_4_df.index)),pic_4_df['zhangting_number'],alpha=0.5, width=0.3, color='red', edgecolor='red', label='涨停股票数量', lw=5)
    bar2 = ax1.bar(np.arange(len(pic_4_df.index))+0.4,pic_4_df['dieting_number'], alpha=0.2, width=0.3, color='green', edgecolor='green', label='跌停股票数量', lw=5)
    ax1.set_xticks(range(0,len(pic_4_df.index),1))#X轴刻度设定 每15天标一个日期
    ax1.set_xticklabels([pic_4_df.index.tolist()[index] for index in ax1.get_xticks()])#标签设置为日期
    for rect in bar1 + bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom',fontsize=15)

    ax1.legend()
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(12)#设置标签字体
    ax1.set_facecolor('white')
    plt.show()


def get_all_market_amount(df):
    all_market_amount_df = df[['date','all_stocks_total_daily_amount']]
    fig = plt.figure(figsize=(16,6))#创建fig对象
    all_market_amount = fig.add_subplot(111)
    all_market_amount_df = all_market_amount_df.set_index('date')
    all_market_amount_df.sort_index(ascending=True,inplace=True)
    # all_market_amount_df.plot(kind='bar',figsize=(30,15),xticks=None)
    bar1 = all_market_amount.bar(np.arange(0, len(all_market_amount_df.index)), all_market_amount_df.all_stocks_total_daily_amount, color='green',label='全市场成交额')
    all_market_amount.set_xticks(range(0,len(all_market_amount_df.index),40))#X轴刻度设定 每15天标一个日期
    all_market_amount.set_xticklabels([all_market_amount_df.index.tolist()[index] for index in all_market_amount.get_xticks()])#标签设置为日期
    for label in all_market_amount.xaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(10)#设置标签字体
    all_market_amount.set_facecolor('white')
    plt.show()


def get_open_auction_number(df):
    open_auction_tick_number_df = df[['date','open_auction_ticker_number','hs_300_return']]
    open_auction_tick_number_df = open_auction_tick_number_df.dropna()
    open_auction_tick_number_df['date'] = open_auction_tick_number_df['date'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d'))
    open_auction_tick_number_df['date'] = pd.to_datetime(open_auction_tick_number_df['date'])
    open_auction_tick_number_df = open_auction_tick_number_df.set_index('date')
    open_auction_tick_number_df = open_auction_tick_number_df.sort_index(ascending=True)
    open_auction_tick_number_df['hs_300_return'] = open_auction_tick_number_df['hs_300_return']+1
    open_auction_tick_number_df.rename(columns={'date':'日期','hs_300_return':'沪深300收益','open_auction_ticker_number':'集合竞价委托笔数'},inplace=True)
    fig = plt.figure(figsize=(16,6))#创建fig对象
    ax1 = fig.add_subplot()
    ax1 = open_auction_tick_number_df['集合竞价委托笔数'].plot(x=open_auction_tick_number_df.index.astype(str),grid=False,label='集合竞价委托笔数',lw=3,fontsize=10)
    ax1.set_facecolor('white')
    ax2 = ax1.twinx()
    ax2 = open_auction_tick_number_df['沪深300收益'].cumprod().plot(x=open_auction_tick_number_df.index.astype(str),grid=False,color='red',lw=3,fontsize=10)
    plt.legend(loc='upper right')
    plt.show()
    