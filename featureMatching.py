import pandas as pd
import hashlib

# 读取两个CSV文件
df1 = pd.read_csv('csv_real.csv')
df2 = pd.read_csv('output_real.csv', encoding='gbk')
df3 = pd.read_csv('output_real_http.csv')

# 两个DataFrame以FlowID进行内连接
merged_df = pd.merge(df1, df2, on='FlowID', how='inner')
merged_df = pd.merge(merged_df, df3, on='FlowID', how='inner')

# 拿掉FlowID
df1 = df1.drop('FlowID', axis=1)

# 添加新的列到文件1
df1['Ciphersuite'] = merged_df['Ciphersuite']
df1['PublickeyLength'] = merged_df['PublickeyLength']
df1['IssuerOrganizationName'] = merged_df['IssuerOrganizationName']
df1['IssuerCommonName'] = merged_df['IssuerCommonName']
df1['SubjectOrganizationName'] = merged_df['SubjectOrganizationName']
df1['SubjectCommonName'] = merged_df['SubjectCommonName']
df1['Host'] = merged_df['Host']
df1['Referer'] = merged_df['Referer']

# 字段去除空格
df1.columns = df1.columns.str.replace(' ', '')

# 空值设为0
df1 = df1.fillna(0)

# Ciphersuite转换成十进制数
df1['Ciphersuite'] = df1['Ciphersuite'].astype(str).apply(lambda x: int(x, 16))

# 字符串量取Hash转数字
df1['IssuerOrganizationName'] = df1['IssuerOrganizationName'].astype(str).\
    apply(lambda x: int(hashlib.sha1(x.encode('utf-8')).hexdigest()[6:10], 16))
df1['IssuerCommonName'] = df1['IssuerCommonName'].astype(str).\
    apply(lambda x: int(hashlib.sha1(x.encode('utf-8')).hexdigest()[6:10], 16))
df1['SubjectOrganizationName'] = df1['SubjectOrganizationName'].astype(str).\
    apply(lambda x: int(hashlib.sha1(x.encode('utf-8')).hexdigest()[6:10], 16))
df1['SubjectCommonName'] = df1['SubjectCommonName'].astype(str).\
    apply(lambda x: int(hashlib.sha1(x.encode('utf-8')).hexdigest()[6:10], 16))
df1['Host'] = df1['Host'].astype(str).\
    apply(lambda x: int(hashlib.sha1(x.encode('utf-8')).hexdigest()[28:32], 16))
df1['Referer'] = df1['Referer'].astype(str).\
    apply(lambda x: int(hashlib.sha1(x.encode('utf-8')).hexdigest()[28:32], 16))


# 获取要移动的列
column_to_move = df1['Label']
# 删除要移动的列
df1 = df1.drop('Label', axis=1)
# 将要移动的列添加到最后
df1['Label'] = column_to_move


# 输出结果到新的CSV文件
df1.to_csv('result_real.csv', index=False, encoding='gbk')