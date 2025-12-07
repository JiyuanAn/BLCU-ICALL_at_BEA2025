import os
import json

# 切分数量设置
DVE_COUNT_LIST = 20

# 加载JSON文件
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# 分割数据集
def split_dataset(data, ratio):
    train_data = data[:int(len(data) * ratio)]
    test_data = data[int(len(data) * ratio):]
    return train_data, test_data

def main():
    # 设置文件路径
    path_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path_input = os.path.join(path_base, "data", "mrbench_v3_devset.json")
    path_output_base = os.path.join(path_base, "data")
    print(f"path_input: {path_input}")
    print(f"path_output: {path_output_base}")

    # 加载数据
    data = load_json(path_input)
    # Dev set
    data_partitioned = data[:DVE_COUNT_LIST]
    with open(os.path.join(path_output_base, f"mrbench_v3_devset_dev.json"), "w") as f:
        json.dump(data_partitioned, f, indent=4, ensure_ascii=False)
    # Train set
    data_partitioned = data[DVE_COUNT_LIST:]
    with open(os.path.join(path_output_base, f"mrbench_v3_devset_train.json"), "w") as f:
        json.dump(data_partitioned, f, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    main()
