import os
import json

SYSTEM_PROMPT = "You are a Senior Teaching Supervisor."
INSTRUCTION = ""
PROMPT_TEMPLATE = """Is it clear from the tutor's latest reply what the student should do next?
- A: Yes (the response provides clear suggestions on what the student should do next)
- B: To some extent (the response indicates that something needs to be done, but it is not clear what exactly that is)
- C: No (the response does not suggest any action on the part of the student (e.g., it simply reveals the final answer))

* Conversation History: “{history}”

* Teacher's Latest Reply: “{tutor_response}”
"""
DIMENSIONS = "Actionability"
ANNOTATION_DICT_New = {"Yes": 'A', "To some extent": 'B', "No": 'C'}

def load_data(path_input):
    with open(path_input, "r") as f:
        data = json.load(f)
    return data


def process_data(data_raw):
    data_processed = []
    for item in data_raw:
        history = item['conversation_history']
        for model in item['tutor_responses']:
            response = item['tutor_responses'][model]['response']
            annotation = item['tutor_responses'][model]['annotation'][DIMENSIONS]
            annotation = "Classification: " + ANNOTATION_DICT_New[annotation]
            data_item = {}
            data_item["system"] = SYSTEM_PROMPT
            # data_item["instruction"] = INSTRUCTION
            data_item["input"] = PROMPT_TEMPLATE.format(history=history, tutor_response=response)
            data_item["output"] = annotation
            data_processed.append(data_item)
    return data_processed


def main():
    # 设置文件路径
    path_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_input = os.path.join(path_base, "data", "mrbench_v3_devset_dev.json")
    path_output = os.path.join(path_base, "res", "mrbench_v3_devset_T4_V02_dev.json")

    # 读取数据文件
    data_raw = load_data(path_input)
    # print(data_raw)

    # 处理数据
    data_processed = process_data(data_raw)
    # print(data_processed)

    # 保存数据
    with open(path_output, "w") as f:
        json.dump(data_processed, f, indent=4)


if __name__ == "__main__":
    main()




