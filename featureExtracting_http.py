import csv
import os

from scapy.layers.http import HTTPRequest
from scapy.utils import rdpcap
from scapy.layers.inet import TCP, IP


def process_pcap_file(pcap_filepath, csv_writer):
    # 使用scapy的rdpcap函数读取pcap文件
    packets = rdpcap(pcap_filepath)

    # 遍历每个数据包
    for packet in packets:
        # 如果是TCP数据包并且包含HTTP协议
        if TCP in packet and packet.haslayer(HTTPRequest):
            # 提取相关字段
            protocol = 6
            src_ip = packet[IP].src
            src_port = packet[TCP].sport
            dst_ip = packet[IP].dst
            dst_port = packet[TCP].dport
            host = packet[HTTPRequest].Host
            referer = packet[HTTPRequest].fields['Referer'].decode('utf-8') if 'Referer' in packet[HTTPRequest].fields else ''
            FlowID = str(src_ip) + '-' + str(dst_ip) + '-' + str(src_port) + '-' + str(dst_port) + '-' + str(protocol)

            # 写入CSV文件
            csv_writer.writerow([FlowID, protocol, src_ip, src_port, dst_ip, dst_port, host, referer])


def walk_pcap_folder(folder_path, output_csv_file):
    # 遍历目录下的所有文件和子目录
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            pcap_filepath = os.path.join(root, filename)
            # 处理pcap文件
            with open(output_csv_file, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                process_pcap_file(pcap_filepath, csv_writer)


if __name__ == '__main__':
    # 遍历当前目录下的所有pcap文件，并将结果保存到csv文件中
    output_csv_file = 'output_real_http.csv'
    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['FlowID', 'Protocol', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Host', 'Referer'])
    walk_pcap_folder(r'D:\csv_real\pcap', output_csv_file)
