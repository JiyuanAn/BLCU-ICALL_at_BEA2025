import json

DIMENSION = "actionability"
ANNOTATION_DICT_New = {2: "Yes", 1: "To some extent", 0: "No"}

# 读取原始JSON文件
def load_raw_json_file(path_raw):
    with open(path_raw, 'r', encoding='utf-8') as file:
        data_raw = json.load(file)
    return data_raw

# 读取结果JSON文件
def load_res_jsonl_file(path_res):
    data_res = []
    with open(path_res, 'r', encoding='utf-8') as file:
        for line in file:
            data_res.append(json.loads(line))
    return data_res

# 合并数据
def merge_data(data_raw, data_res):
    data_merged = []
    for item_data in data_raw:
        print(item_data)
        conversation_id = item_data['conversation_id']
        tutor_name_list = item_data['tutor_responses'].keys()
        for tutor_name in tutor_name_list:
            item_id = conversation_id + "#_#" + DIMENSION + "#_#" + tutor_name
            for item_res in data_res:
                if item_res['item_id'] == item_id:
                    item_data['tutor_responses'][tutor_name]['annotation'] = {}
                    item_data['tutor_responses'][tutor_name]['annotation']['Actionability'] = ANNOTATION_DICT_New[item_res['res']]
        data_merged.append(item_data)
    return data_merged

# 写入提交文件
def write_submit_file(data_merged, path_submit):
    with open(path_submit, 'w', encoding='utf-8') as file:
        json.dump(data_merged, file, ensure_ascii=False, indent=4)


def main():
    # 设置文件路径
    path_raw = '/home/wmy2024/workspace/BEA-20/Inference/data/mrbench_v3_testset.json'
    path_res = '/home/wmy2024/workspace/BEA-20/Inference/res/merge/merged_results_T1-NP-2_filtered.jsonl'
    path_submit = './predictions.json'
    # 读取原始JSON文件
    data_raw = load_raw_json_file(path_raw)
    # 读取结果JSON文件
    data_res = load_res_jsonl_file(path_res)
    # 合并数据
    data_merged = merge_data(data_raw, data_res)
    # 写入提交文件
    write_submit_file(data_merged, path_submit)

if __name__ == "__main__":
    main()