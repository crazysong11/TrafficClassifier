import csv
import subprocess
import os

try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
except BaseException as exp:
    raise BaseException(
        'Please install cryptography library: pip3 install cryptography -i https://mirrors.aliyun.com/pypi/simple/')


def bytes_to_string(bytes):
    return str(bytes, 'utf-8')


def x509name_to_json(x509_name):
    json = {}
    for attribute in x509_name:
        name = attribute.oid._name
        value = attribute.value
        json[name] = value
    return json


def x509_parser(cert_hex):
    try:
        cert = bytes.fromhex(cert_hex.replace(':', ''))
        cert = x509.load_der_x509_certificate(cert, default_backend())
        rst = {
            'issuer': x509name_to_json(cert.issuer),
            'subject': x509name_to_json(cert.subject),
        }
        return rst
    except Exception as e:
        print(f"解析证书时出错: {e}")
        return None


def extract_pcap_features(pcap_file):
    # 调用tshark命令行工具
    cmd = ['tshark', '-r', pcap_file, '-T', 'fields', '-e', 'frame.number',
           '-e', 'ip.src', '-e', 'ip.dst', '-e', 'tcp.srcport', '-e', 'tcp.dstport',
           '-e', 'ip.proto', '-e', 'tls.handshake.ciphersuite', '-e', 'tls.handshake.certificate',
           '-e', 'tls.handshake.client_point_len']
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        output = result.stdout.strip().split('\n')
        features = [line.split('\t') for line in output]
        return features
    else:
        print(f"提取特征时出错: {result.stderr}")
        return None


with open('tls_real.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['FlowID', 'SourceIP', 'DestinationIP', 'SourcePort',
                     'DestinationPort', 'Protocol', 'Ciphersuite',
                     'PublickeyLength', 'IssuerOrganizationName',
                     'IssuerCommonName', 'SubjectOrganizationName',
                     'SubjectCommonName'])

# 定义要读取的文件夹路径
folder_path = r'D:\csv_real\pcap'

# 获取文件夹下的所有文件
files = os.listdir(folder_path)

# 遍历每个文件
for file_name in files:
    
    pcap_file = os.path.join(folder_path, file_name)
    features = extract_pcap_features(pcap_file)

    if features:
        try:
            length = len(features)
            # 遍历列表，对每一行进行处理
            for i in range(length):
                while len(features[i]) < 9:
                    features[i].append('')

            for i in range(length):
                if features[i][5] != '':
                    for j in range(length):
                        features[j][5] = features[i][5]
                    break
            flag = 0
            for i in range(length):
                if features[i][6] != '':
                    if flag == 1:
                        for j in range(length):
                            features[j][6] = features[i][6]
                        break
                    flag = 1
            for i in range(length):
                if features[i][7] != '':
                    for j in range(length):
                        features[j][7] = features[i][7]
                    break
            for i in range(length):
                if features[i][8] != '':
                    for j in range(length):
                        features[j][8] = features[i][8]
                    break
        except:
            raise Exception("no")

        # 输出到CSV文件
        with open('tls_real.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for feature in features:
                try:
                    if feature[5] != '' or feature[6] != '' or feature[7] != '' or feature[8] != '':
                        # 解析ciphersuites
                        ciphersuites = feature[6].split(',')
                        length = len(ciphersuites)
                        for i in range(length):
                            ciphersuites[i] = ciphersuites[i][2:]
                        ciphersuites = ''.join(ciphersuites)
                        feature[6] = ciphersuites
                        if feature[7] != '':
                            # 解析证书
                            certs = feature[7].split(',')
                            cert_json = x509_parser(certs[0])
                            if 'organizationName' in cert_json['issuer']:
                                feature.append(cert_json['issuer']['organizationName'])
                            else:
                                feature.append('')
                            if 'commonName' in cert_json['issuer']:
                                feature.append(cert_json['issuer']['commonName'])
                            else:
                                feature.append('')
                            if 'organizationName' in cert_json['subject']:
                                feature.append(cert_json['subject']['organizationName'])
                            else:
                                feature.append('')
                            if 'commonName' in cert_json['subject']:
                                feature.append(cert_json['subject']['commonName'])
                            else:
                                feature.append('')
                            del feature[7]
                        else:
                            feature.append('')
                            feature.append('')
                            feature.append('')
                            feature.append('')
                            del feature[7]
                        feature[0] = feature[1] + '-' + feature[2] + '-' + \
                                     feature[3] + '-' + feature[4] + '-' + feature[5]
                        writer.writerow(feature)
                except:
                    raise Exception("Error!")

print("已将结果输出到tls_real.csv文件中。")

import pandas as pd

# 读取CSV文件
csv_file = 'tls_real.csv'
df = pd.read_csv(csv_file, encoding='gbk')

# 去除重复行
df.drop_duplicates(inplace=True)

# 保存结果到新的CSV文件
output_file = 'output_real.csv'
df.to_csv(output_file, index=False, encoding='gbk')

print("去重完成。")
