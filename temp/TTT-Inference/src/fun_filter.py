import os
import json
from collections import Counter

path_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path_base)
input_file_path = os.path.join(path_base, 'res', 'merge', 'merged_results_T1-NP-2.jsonl')
output_file_path = os.path.join(path_base, 'res', 'merge', 'merged_results_T1-NP-2_filtered.jsonl')

# 打开输入文件和输出文件
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    # 逐行读取输入文件
    for line in infile:
        # 解析JSON字符串为Python字典
        item = json.loads(line)
        # 获取label列表并统计元素出现次数
        label_counts = Counter(item['label'])
        # 找出出现次数大于等于3的元素
        res = [k for k, v in label_counts.items() if v >= 1]
        # 如果没有符合条件的元素，则设置为None
        if not res:
            res = None
        else:
            # 如果有多个符合条件的元素，只取第一个
            res = res[0]
        # 为item添加res项
        item['res'] = res
        # 将更新后的item写入输出文件
        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')