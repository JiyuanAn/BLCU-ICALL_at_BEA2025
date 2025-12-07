import os
import re
import csv
import json

DIMENSION_LIST = ["actionability"] #["mistake_identification", "mistake_location", "providing_guidance", "actionability"]
ANNOTATION_DICT = {"Yes": 2, "To some extent": 1, "No": 0}
Error_value = -2

# 读取原始数据
def load_json_raw(path_input_raw):
    item_name_list = []
    with open(path_input_raw, 'r', encoding='utf-8') as f:
        data_raw = json.load(f)
    for item in data_raw:
        conversation_id = item['conversation_id']
        Tutor_name_list = item['tutor_responses'].keys()
        item_name_list.append({'conversation_id': conversation_id, 'Tutor_name_list': Tutor_name_list})
    return item_name_list

# 读取预测数据
def load_json_pred(path_input_pred):
    item_response_list = []
    with open(path_input_pred, 'r', encoding='utf-8') as f:
        data_pred = json.load(f)
    for item in data_pred:
        conversation_id = item['conversation_id']
        annotation_pred_list = []
        for tutor_name in item['tutor_responses']:
            for dimension in DIMENSION_LIST:
                annotation_pred_str = item['tutor_responses'][tutor_name]['annotation_pred'][dimension]['response']
                label = -1
                if annotation_pred_str != "" and annotation_pred_str != None:
                    
                    text = annotation_pred_str.strip()
                    match = re.search(r'[ABC]', text)
                    if match:
                        print(match.group())
                        if match.group() == 'A':
                            label = 2
                        elif match.group() == 'B':
                            label = 1
                        else:
                            label = 0
                    else:
                        #label = list(ANNOTATION_DICT.values())[-1]
                        label = Error_value
                    
                else:
                    #label = list(ANNOTATION_DICT.values())[-1]
                    label = Error_value
                item_id = item['conversation_id'] + "#_#" + dimension + "#_#" + tutor_name
                item_response_list.append({'item_id': item_id, 'label': label})
    print(len(item_response_list))
    return item_response_list

# 合并数据
def merge_data(data_raw, data_pred):
    data_merged = {}
    for item in data_raw:
        conversation_id = item['conversation_id']
        Tutor_name_list = item['Tutor_name_list']
        for dimension in DIMENSION_LIST:
            for tutor_name in Tutor_name_list:
                item_id = conversation_id + "#_#" + dimension + "#_#" + tutor_name
                for model_item in data_pred:
                    for item_pred in model_item:
                        if item_pred['item_id'] == item_id:
                            if item_id not in data_merged:
                                data_merged[item_id] = []
                            data_merged[item_id].append(item_pred['label'])
    return data_merged

# 保存合并后的数据
def save_csv(data_merged, path_output):
    with open(path_output + ".csv", 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['item_id', 'label'])
        for item_id, label in data_merged.items():
            writer.writerow([item_id, label])

# 保存合并后的数据
def save_jsonl(data_merged, path_output):
    with open(path_output + ".jsonl", 'w', encoding='utf-8') as f:
        for item_id, label in data_merged.items():
            f.write(json.dumps({'item_id': item_id, 'label': label}, ensure_ascii=False) + '\n')

# 主函数
def main():
    # 设置命令行参数解析
    path_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(path_base)
    path_input_raw = os.path.join(path_base, 'data', 'mrbench_v3_testset.json')
    path_input_pred_base = os.path.join(path_base, 'res')
    path_output = os.path.join(path_base, 'res', 'merge', 'merged_results_T1-NP-2')
    # 读取原始数据
    data_raw = load_json_raw(path_input_raw)
    # 读取预测数据
    file_list = os.listdir(path_input_pred_base)
    file_list = [file for file in file_list if file.endswith('.json')]
    file_list = sorted(file_list)
    print(file_list)
    data_pred = []
    for file in file_list:
        path_input_pred = os.path.join(path_input_pred_base, file)
        data_pred.append(load_json_pred(path_input_pred))
    # 合并数据
    data_merged = merge_data(data_raw, data_pred)
    # 保存合并后的数据
    save_csv(data_merged, path_output)
    save_jsonl(data_merged, path_output)

if __name__ == '__main__':
    main()