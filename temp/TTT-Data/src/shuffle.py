import json
import random

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 打乱顺序
def shuffle_data(data):
    if isinstance(data, list):
        random.shuffle(data)
    else:
        print("JSON数据不是列表格式，无法打乱顺序。")
    return data

# 将数据写入新的JSON文件
def write_to_file(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"打乱顺序后的数据已写入文件：{output_file_path}")

# 主程序
if __name__ == "__main__":
    input_file_path = "../res/mrbench_v3_devset_T4_V01_train.json"  # 输入的JSON文件路径
    output_file_path = "../res/mrbench_v3_devset_T4_V01_train_shuffled.json"  # 输出的JSON文件路径

    # 读取数据
    data = read_json_file(input_file_path)
    # 打乱顺序
    shuffled_data = shuffle_data(data)
    # 写入新文件
    write_to_file(shuffled_data, output_file_path)