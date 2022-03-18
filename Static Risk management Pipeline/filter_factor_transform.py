import pandas as pd
from smartbeta.smartfactor import SmartFactor
from smartbeta.ai_factor import AiFactor
from data_provider.nestlib.market_info import Frequency

class FilterFactor(AiFactor):
    '''
    该类负责将一个因子对另一个因子进行Union操作并留下共有部分的因子
    请使用工具函数generate_filter_factor
    '''

    def _x_rules(self):
        """
        在该方法中指定因子预处理方式
        
        更多方法，参见因子合成文档中的因子标准化方法
        """
        return []

    def _build_ai_model(self, dateTime, training_set):
        """
        按指定frequency，滚动生成复合因子数据
        
        Parameters
        -----------
        dateTime: datetime类型
            当前时间
        trainint_set: pd.DataFrame
            训练集
        Return
        -------
        pd.Series类型，key为securityId, 值为factor value
        """
        ret = self.ff._get_training_XY(dateTime, training_set)
        
        target_name=self._factor_param['subFactors'][-1]['factor_name']
        source_name=self._factor_param['subFactors'][0]['factor_name']
        
        df_factor=ret[2].reset_index().set_index('security_code')
        if 'filter_source_percentile' in self._factor_param:
            if self._factor_param['filter_source_percentile'] != None:
                filter_percentile=self._factor_param['filter_source_percentile']
                sort_asc=False
                if 'filter_source_ascending' in self._factor_param:
                    sort_asc=self._factor_param['filter_source_ascending']
                df_factor=df_factor.sort_values(source_name, ascending=sort_asc)
                the_count=int(df_factor.shape[0]*filter_percentile)
                df_factor=df_factor.iloc[:the_count]
        
        result = df_factor.loc[:,target_name]
        return result
    
def generate_filter_factor(filter_factor_name, source_factor_name, 
                            new_factor_name, from_dt, to_dt, filter_percentile=None, filter_ascending=False, echo=False, is_external=False):
    '''
    This function would generate a new factor which is the source_factor but filtered by another.
    @param filter_factor_name: the factor name to filter the source
    @param source_factor_name: the source factor name whose factor value would be reserved.
    @param new_factor_name: the generated new factor name.
    @param filter_percentile: a value between 0~1, the percentile to filter the filter factor. 
                              It allows to reserve only top factors of filter_factor.
                              Set this value to None if no percentile filter needed       
                                     
    @param filter_ascending: True to reserve only head factor values of on filter percentile,
                             False to reserve only tail factor values of filter percentile

    @param is_external: set to True if it is an industry factor or index factor
                             
    @return: No return value. Check private factor list for the generated factor.
    '''
    
    subFactors = [
        {'factor_name':filter_factor_name, 'factor_direction':1},
        {'factor_name':source_factor_name, 'factor_direction':1},
    ]

    # 指定复合因子参数
    factor_parameters={"subFactors": subFactors,
                       "frequency": 'daily',
                       "lagCycleNumber": 0,
                       "use_ranks_as_x": False,
                       "use_ranks_as_y": False,
                       "class_percentile": False,
                       "class_remove_neural": False,
                       "treat_na": 'drop', 
                       "include_latest": False,
                       "filter_source_percentile": filter_percentile,
                       "filter_source_ascending": filter_ascending,
                       'dailyreturn_factor_name':'dailyreturn',
                       'source_factorname':,
                       'filter_factor_dict':}

    ab = FilterFactor(
        factor_name=new_factor_name, # 复合因子的名称
        tickers='A', # 复合因子对应的股票池
        factor_parameters=factor_parameters,
    )

    # 是否使用cache
    ab.set_use_factor_cache(False)
    if is_external is True:
        ab.set_factor_external(True)
    #从数据库清空因子，以便重新录入
    ab.generate_factor_and_store(from_dt, to_dt, echo=echo)
    print('因子合成完毕，已成功入库!')