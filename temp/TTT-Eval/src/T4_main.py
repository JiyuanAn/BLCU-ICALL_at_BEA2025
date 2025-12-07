import os
import re
import json
import argparse

ANNOTATION_DICT_Past = {"Yes": 0, "To some extent": 1, "No": 2}
ANNOTATION_DICT_New = {"Yes": 2, "To some extent": 1, "No": 0}
DIMENSION_DICT = {"mistake_identification": "Mistake_Identification", "mistake_location": "Mistake_Location", "providing_guidance": "Providing_Guidance", "actionability": "Actionability"}
CLASSES = [0, 1, 2]
DIMENSION = "actionability"
ERROR_CLASS = {"01": 0, "02": 1, "10": 2, "12": 0, "20": 1, "21": 2}

# 读取JSON文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 处理数据
def process_data(data, ANNOTATION_DICT):
    y_true_dict = {"Mistake_Identification": [], "Mistake_Location": [], "Providing_Guidance": [], "Actionability": []}
    y_pred_dict = {"Mistake_Identification": [], "Mistake_Location": [], "Providing_Guidance": [], "Actionability": []}
    for index in range(len(data)):
        item = data[index]
        for model in item['tutor_responses']:
            y_true = 0
            y_pred = 0
            for dimension in item['tutor_responses'][model]['annotation']:
                label = item['tutor_responses'][model]['annotation'][dimension]
                y_true_dict[dimension].append(ANNOTATION_DICT[label])
                for key, value in DIMENSION_DICT.items():
                    if value == dimension:
                        dimension = key
                if dimension == DIMENSION:
                    y_true = ANNOTATION_DICT[label]
            for dimension in item['tutor_responses'][model]['annotation_pred']:
                # y_pred_dict[DIMENSION_DICT[dimension]].append(item['tutor_responses'][model]['annotation_pred'][dimension])
                label = -1
                
                text = item['tutor_responses'][model]['annotation_pred'][dimension]['response'].strip()
                print(text)
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
                    label = list(ANNOTATION_DICT.values())[-1]


                y_pred_dict[DIMENSION_DICT[dimension]].append(label)
                y_pred = label
                if y_true != y_pred and DIMENSION == dimension:
                    print(f"Index: {index+1}, Dimension: {dimension}, Model: {model}, y_true: {y_true}, y_pred: {y_pred}")
                    ERROR_CLASS[f'{y_true}{y_pred}'] += 1
    return y_true_dict, y_pred_dict, ERROR_CLASS


# 计算准确率
def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total = len(y_true)
    return correct / total

# 计算准确率
def calculate_accuracy_lenient(y_true, y_pred, ANNOTATION_DICT):
    y_true = [list(ANNOTATION_DICT.values())[0] if x == 1 else x for x in y_true]
    y_pred = [list(ANNOTATION_DICT.values())[0] if x == 1 else x for x in y_pred]
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total = len(y_true)
    return correct / total

# 计算宏观F1
def calculate_macro_f1(y_true, y_pred, classes):
    # 为每个类别计算TP, FP, FN
    # print(y_true)
    # print(y_pred)
    tp = {cls: 0 for cls in classes}
    fp = {cls: 0 for cls in classes}
    fn = {cls: 0 for cls in classes}
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    # 计算每个类别的F1
    f1_scores = []
    for cls in classes:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) != 0 else 0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_scores.append(f1)
        print(f"      Class: {cls}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    # 宏观F1是所有类别F1的平均值
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1

# 计算宏观F1
def calculate_macro_f1_lenient(y_true, y_pred, classes, ANNOTATION_DICT):
    y_true = [list(ANNOTATION_DICT.values())[0] if x == 1 else x for x in y_true]
    y_pred = [list(ANNOTATION_DICT.values())[0] if x == 1 else x for x in y_pred]
    # 为每个类别计算TP, FP, FN
    tp = {cls: 0 for cls in classes}
    fp = {cls: 0 for cls in classes}
    fn = {cls: 0 for cls in classes}
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    # 计算每个类别的F1
    f1_scores = []
    for cls in classes:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) != 0 else 0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_scores.append(f1)
    # 宏观F1是所有类别F1的平均值
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1

# 主函数
def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Calculate indicators for response evaluation')
    parser.add_argument('input_file', type=str, help='Path to the input JSON file')
    parser.add_argument('rating_order', type=str, help='Rating order')
    args = parser.parse_args()
    # 设置评分顺序
    if args.rating_order == "Past":
        ANNOTATION_DICT = ANNOTATION_DICT_Past
    elif args.rating_order == "New":
        ANNOTATION_DICT = ANNOTATION_DICT_New
    else:
        raise ValueError(f"Invalid rating order: {args.rating_order}")
    # 读取JSON文件
    data = load_json(args.input_file)
    # 处理数据
    y_true_dict, y_pred_dict, ERROR_CLASS = process_data(data, ANNOTATION_DICT)
    classes = CLASSES
    print("------------------------------")
    print(ERROR_CLASS)
    print("------------------------------")
    for dimension in y_true_dict.keys():
        y_true = y_true_dict[dimension]
        y_pred = y_pred_dict[dimension]
        # 计算严格的准确率和宏观F1
        accuracy = calculate_accuracy(y_true, y_pred) # 计算准确率
        macro_f1 = calculate_macro_f1(y_true, y_pred, [0, 1, 2]) # 计算宏观F1
        print(f"Dimension: {dimension}, Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}, length: {len(y_true)}-{len(y_pred)}")
        # 计算宽松的准确率和宏观F1
        accuracy_lenient = calculate_accuracy_lenient(y_true, y_pred, ANNOTATION_DICT) # 计算准确率
        macro_f1_lenient = calculate_macro_f1_lenient(y_true, y_pred, [0, 2], ANNOTATION_DICT) # 计算宏观F1
        print(f"Dimension: {dimension}, Lenient Accuracy: {accuracy_lenient:.4f}, Lenient Macro F1: {macro_f1_lenient:.4f}, length: {len(y_true)}-{len(y_pred)}")


if __name__ == "__main__":
    main()