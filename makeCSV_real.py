import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import random

np.set_printoptions(suppress=True, precision=20, threshold=10, linewidth=40)  # np禁止科学计数法显示
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # pd禁止科学计数法显示

# 定义要读取的文件夹路径
folder_path = r'D:\csv_real\CSV'

# 遍历文件夹下所有文件名，筛选出所有 CSV 文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 读取所有CSV文件并存储到一个列表中
dfs = []
for csv_file in csv_files:
    csv_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(csv_path)
    dfs.append(df)

# 对所有CSV文件进行拼接
df_all = pd.concat(dfs)

# 字段去除空格
df_all.columns = df_all.columns.str.replace(' ', '')

df_all = df_all.rename(columns={'SrcIP':'SourceIP', 'DstIP':'DestinationIP',
                       'SrcPort':'SourcePort', 'DstPort':'DestinationPort'})

# 修正FlowID格式
df_all['FlowID'] = df_all['SourceIP'].astype(str) + '-' + df_all['DestinationIP'].astype(str) + \
                    '-' + df_all['SourcePort'].astype(str) + '-' + df_all['DestinationPort'].astype(str) + \
                    '-' + df_all['Protocol'].astype(str)

# 筛选保留字段
df_all = df_all[['FlowID', 'DestinationPort', 'Protocol', 'TotFwdPkts', 'TotBwdPkts',
                 'TotLenFwdPkts', 'TotLenBwdPkts', 'FlowDuration',
                 'FwdPktLenMax', 'FwdPktLenMin', 'FwdPktLenMean',
                 'BwdPktLenMax', 'BwdPktLenMin', 'BwdPktLenMean',
                 'FlowIATMean', 'FlowIATStd', 'FlowIATMax', 'FlowIATMin', 'FwdIATTot', 'BwdIATTot',
                 'SYNFlagCnt', 'PSHFlagCnt', 'ACKFlagCnt',
                 'URGFlagCnt', 'Label']]


# 根据条件修改Label的值
df_all.loc[df_all['Label'] != 1, 'Label'] = 0

df_all.drop_duplicates(inplace=True)  # 使用drop_duplicates去重，inplace=True对原数据集进行替换
df_all.reset_index(drop=True, inplace=True)  # 删除数据后，恢复索引

df_all = df_all.fillna(0)

df_all.to_csv('csv_real.csv', index=False)