# 项目结构

- featureExtracting_http.py：用于提取pcap中的HTTP特征
- featureExtracting_tls.py：用于提取pcap中的TLS特征
- makeCSV_real.py：对流元特征进行加工处理
- featureMatching.py：用于将提取的HTTP和TLS特征匹配添加到原有特征之后并进行加工处理，生成完整特征文件数据集
- net.py：CNN所需模型
- CNN.py：CNN深度学习实现代码
- RF_Plot.py：随机森林实现代码
- XG_Plot.py：XGBoost实现代码
- SVM_Plot.py：SVM实现代码
- Stacking_Plot.py：Stacking集成学习实现代码