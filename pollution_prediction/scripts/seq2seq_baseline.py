#!/usr/bin/env python
# coding=utf-8

'''
    author:fengjiaxin
    最简单的seq2seq_baseline
    其中预测当天小时的污染程度，但是有一些自变量是已知的
    并且前几天的污染程度也可以当作一些特征，问题的关键是如何构建这些特征进行训练
    应用keras ,而不是使用tensorflow 的keras
'''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import date, timedelta,datetime
import gc
from sklearn import metrics
from keras import Model
from keras.layers import Input,Lambda,concatenate,GRU,Dense,Conv1D,Reshape,Embedding
from keras.utils import Sequence
import math
import logging
logging.basicConfig(level=logging.INFO)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




'''
整理思路：
1.首先需要预测的是pollution
2.需要预测的是后7天的每个小时的污染程度
3.先假定根据前一个月的数据来预测后7天的数据


有个理解错误的地方
timesetps时间步的含义是t时刻的value 与 t-timesteps ~ t-1 的value相关
在decoder的过程中，需要不断更新seq

'''

def create_xy_span(df, pred_start, timesteps,periods,is_train=True):
    '''
        function:首先说一下需求，然后才能更好的了解函数的功能
        需求:该时刻的pollution可能和60天前的pollution相关，需要将60天前的pollution提取出来，作为该时刻的特征
        timesteps代笔时间步，我们需要构建timesteps个时刻的的特征
        periods代表需要预测的时间段长度        
    '''
    X = df[pd.date_range(pred_start-timedelta(hours=timesteps), pred_start-timedelta(hours=1),freq='H')].values
    if is_train: y = df[pd.date_range(pred_start, periods = periods,freq='H')].values
    else: y = None
    return X, y


def create_dataset_base(df, dew_df, timesteps,periods, pred_start, is_train):
    '''
        function:创建数据集合
        df:污染数据
        dew_df:露点数据
        timesteps:时间步
        pred_start:开始预测的时间点
        periods:后续预测的时间段
        is_train:是否是训练数据
    '''

    # 首先可以获取最基本的pollution数据
    # X:[1,timesteps] y:[1,periods]
    X, y = create_xy_span(df, pred_start, timesteps,periods,is_train)
    
    # dew 包含训练和测试的露点数据
    # dew:[1,timesteps + periods]
    dew = dew_df[pd.date_range(pred_start-timedelta(hours=timesteps), periods=timesteps+periods,freq='H')].values
    
    
    # 获取hour特征 hour:[1,timesteps + periods]
    hour = np.tile([d.hour for d in pd.date_range(pred_start-timedelta(hours=timesteps), periods=timesteps+periods,freq='H')],
                          (X.shape[0],1))
    
    # 获取week特征 week:[1,timesteps + periods]
    weekday = np.tile([d.weekday() for d in pd.date_range(pred_start-timedelta(hours=timesteps), periods=timesteps+periods,freq='H')],
                          (X.shape[0],1))
    # 获取day特征 day:[1,timesteps + periods]
    dayOfMonth = np.tile([d.day-1 for d in pd.date_range(pred_start-timedelta(hours=timesteps), periods=timesteps+periods,freq='H')],
                          (X.shape[0],1))
    
    # 将3个月前,91天的pollution作为特征 quater:[1,timesteps + periods]
    #qauaterAgo, _ = create_xy_span(df, pred_start-timedelta(days=91), timesteps+16, False)
    
    # reshape 数据
    X= X.reshape(-1,timesteps,1)
    dew = dew.reshape(-1, timesteps + periods, 1)
    #hour = hour.reshape(-1, timesteps + periods, 1)
    #weekday = weekday.reshape(-1, timesteps + periods, 1)
    #dayOfMonth = dayOfMonth.reshape(-1, timesteps + periods, 1)
    #qauaterAgo = qauaterAgo.reshape(-1, timesteps + periods, 1)

    #return ([X,dew,weekAgo,hour,weekday,dayOfMonth,qauaterAgo],y)
    return ([X,dew,hour,weekday,dayOfMonth],y)

# Create validation and test data
def create_dataset(df, dew_df, timesteps, periods,first_pred_start, is_train=True):
    return create_dataset_base(df, dew_df, timesteps,periods, first_pred_start, is_train)

'''
接下来创建训练数据生成器，每次批量生成数据
在这个例子中，batch_size = 1
n_range 代表生成多少个样本
'''


class DataGenerator(Sequence):
    '''
        generate data for keras
        df,dew_df都是一行数据
        这个比较特殊，只能先设定batch_size = 1
    '''
    def __init__(self,pollution_df, dew_df, timesteps, first_pred_start,n_range, periods, batch_size,shuffle=True):
        '''
            df:pollution数据，列为时间
            dew_df:露点数据，列为时间
            timesteps:时间步
            first_pred_start:需要预测pollution的开始日期
            n_range:代表可以划分为多少个时间段数据
            假设df 列 如下：
            1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10
            timesteps = 3
            periods = 2
            first_pred_start = 1.9
            那么可以确定划分为6份数据
            batch_size:批量数据
        '''
        self.pollution_df = pollution_df
        self.dew_df = dew_df
        self.timesteps = timesteps
        self.first_pred_start = first_pred_start
        self.n_range = n_range
        self.periods = periods
        if batch_size != 1:
            self.batch_size = 1
        else:
            self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
        
    def __len__(self):
        '''
            计算每一个epoch的迭代个数
        '''
        return math.ceil(self.n_range/self.batch_size)
    
    
    def on_epoch_end(self):
        '''
            每次epoch结束后,self.date_indexs是否需要进行打乱
        '''
        self.date_indexs = np.arange(self.n_range)
        if self.shuffle:
            np.random.shuffle(self.date_indexs)
    
    def __getitem__(self,index):
        # 预测的日期
        pred_start = self.first_pred_start - timedelta(hours=int(self.date_indexs[index]))
        return create_dataset_base(self.pollution_df, self.dew_df, self.timesteps,self.periods, pred_start, is_train=True)


    
# Calculate RMSE scores for all 7 days, first 1 days (fror public LB) and 2th-7th days (for private LB) 
def cal_score(Ytrue, Yfit):
    logging.info('all RMSE\t' + str(metrics.mean_squared_error(Ytrue, Yfit)))
    logging.info('public RMSE\t' + str(metrics.mean_squared_error(Ytrue[:,:5], Yfit[:,:5])))
    logging.info('private RMSE\t' + str(metrics.mean_squared_error(Ytrue[:,5:], Yfit[:,5:])))
    
    
def train_model():
    logging.info('--------------------begin construct data------------------')
    raw_df = pd.read_csv('../data/pollution.csv',parse_dates = ['date'])

    '''
    可以看出数据的日期是 从 2010-01-02 00:00:00  到  2014-12-31 23:00:00
    其中说明一下各列代表的含义：
    pollution:PM2.5浓度
    dew:露点
    temp:温度
    press:压力
    wnd_dir:综合风向
    wnd_spd:累计风速
    snow:累计下了几个小时的雪
    rain:累计下了几个小时的雨
    目的：生成两个df
    pollution_df:只有一行，多列，其中列是日期，value是污染pollution
    dew_df:只有一行，多列，其中列是日期，value是露点dew
    '''

    # 生成pollution_df
    raw_df['variable'] = 'pollution'
    pollution_df = raw_df[['date','pollution','variable']].set_index(['variable','date'])[['pollution']].unstack(level=-1)
    pollution_df.columns = pollution_df.columns.get_level_values(1)


    # 生成dew_df
    raw_df['variable'] = 'dew'
    dew_df = raw_df[['date','pollution','variable']].set_index(['variable','date'])[['pollution']].unstack(level=-1)
    dew_df.columns = dew_df.columns.get_level_values(1)
    
    ######################################################################################################################

    timesteps = 30 * 24
    periods = 7 * 24
    # 其中数据的最后一天日期是2014年12月31日

    logging.info('start generate train val test data')

    # 数据生成器
    training_generator = DataGenerator(pollution_df, dew_df, timesteps, datetime(2014,12,11,0),1000, periods,1,shuffle=True)
    #train_data = train_generator(pollution_df, dew_df, timesteps, datetime(2014,12,11,0),periods,200)


    Xval, Yval = create_dataset(pollution_df, dew_df, timesteps, periods,datetime(2014,12,18,0), is_train=True)
    logging.info('end generate train val test data')

    Xtest, Ytest = create_dataset(pollution_df, dew_df, timesteps, periods,datetime(2014,12,25,0), is_train=True)

    del pollution_df, dew_df; gc.collect()





    latent_dim = 10
    # input [X,dew,hour,weekday,dayOfMonth]

    seq_in = Input(shape=(timesteps, 1))

    # 下面这些特征都是已知的
    dew_in = Input(shape=(timesteps+periods, 1))

    # embedding
    hour_in = Input(shape=(timesteps+periods,),dtype='uint8')
    hour_embed_encode = Embedding(24, 6, input_length=timesteps+periods)(hour_in)


    weekday_in = Input(shape=(timesteps+periods,), dtype='uint8')
    weekday_embed_encode = Embedding(7, 4, input_length=timesteps+periods)(weekday_in)

    dayOfMonth_in = Input(shape=(timesteps+periods,), dtype='uint8')
    dayOfMonth_embed_encode = Embedding(31, 7, input_length=timesteps+periods)(dayOfMonth_in)



    # Encoder
    # 截取前timesteps时间步的数据
    encode_slice = Lambda(lambda x: x[:, :timesteps, :])
    # 合并已知的特征向量
    encode_features = concatenate([dew_in, hour_embed_encode, weekday_embed_encode, dayOfMonth_embed_encode], axis=2)

    encode_features = encode_slice(encode_features)

    conv_in =Conv1D(4, 5, padding='same')(seq_in)

    x_encode = concatenate([seq_in, encode_features, conv_in], axis=2)

    encoder = GRU(latent_dim, return_state=True)

    _, h= encoder(x_encode)


    # Connector
    h = Dense(latent_dim, activation='tanh')(h)


    # Decoder
    previous_x = Lambda(lambda x: x[:, -1, :])(seq_in)

    decode_slice = Lambda(lambda x: x[:, timesteps:, :])

    decode_features = concatenate([dew_in, hour_embed_encode, weekday_embed_encode, dayOfMonth_embed_encode], axis=2)

    decode_features = decode_slice(decode_features)


    decoder = GRU(latent_dim, return_state=True, return_sequences=False)
    decoder_dense = Dense(1, activation='relu')

    # 需求 x:[batch_size,timesteps,feature_size] 想获取第i个时间步的特征  t时刻的特征[batch_size,1,feature_size]
    slice_at_t = Lambda(lambda x: x[:,i:i+1,:])

    for i in range(periods):

        previous_x = Reshape((1,1))(previous_x)

        features_t = slice_at_t(decode_features)

        decode_input = concatenate([previous_x, features_t], axis=2)

        output_x, h = decoder(decode_input, initial_state=h)

        output_x = decoder_dense(output_x)

        # gather outputs
        if i == 0: decoder_outputs = output_x
        elif i > 0: decoder_outputs = concatenate([decoder_outputs, output_x])

        previous_x = output_x


    logging.info('model begin define')

    # [X,dew,hour,weekday,dayOfMonth],y
    model = Model([seq_in, dew_in, hour_in, weekday_in,dayOfMonth_in], decoder_outputs)

    logging.info('model begin compile')

    model.compile(optimizer='adam', loss='mean_squared_error')

    logging.info('model begin train')

    history = model.fit_generator(training_generator, workers=3,use_multiprocessing=True, epochs=5, verbose=2,validation_data=(Xval, Yval))
    logging.info('model train success')

    model.save('../model/seq2seq_baseline_model_all.h5')

    logging.info('model save success')
    
    logging.info('evaluate val data')
    val_pred = model.predict(Xval)
    cal_score(Yval, val_pred)
    
    logging.info('evaluate test data')
    test_pred = model.predict(Xtest)
    cal_score(Ytest, test_pred)
    
    logging.info('process all end!!!')
    
if __name__ == '__main__':
    train_model()
    
    
