import os
import json

SYSTEM_PROMPT = "You are a Senior Teaching Supervisor."
INSTRUCTION = ""
PROMPT_TEMPLATE = """# Objective: Evaluate the quality of a teacher's latest response within the context of an ongoing conversation with a student. Your evaluation must be based solely on the provided information and result in structured feedback and a grade classification.

# Inputs:
* **Evaluation Indicators:** “{definition}”
* **Grading Criteria:** {rubric}
* **Conversation History:** “{history}”
* **Teacher's Latest Reply:** “{tutor_response}”

# Instructions:
1.  **Analyze:** Carefully review the **Teacher's Latest Reply** in the context of the **Conversation History**.
2.  **Evaluate:** Assess the **Teacher's Latest Reply** strictly against each point listed in the **Evaluation Indicators**.
3.  **Assign Grade:** Based on your step-by-step evaluation and the provided **Grading Criteria**, determine the appropriate classification (A, B, or C).
4.  **Format Output:** Present your response *only* in the following format, without any additional introductory or concluding remarks:
`Classification: (A, B, or C)`
"""
DEFINITION = "Does the tutor's response accurately point to a genuine mistake and its location?"
RUBRIC = """
A: Yes (the tutor clearly points to the exact location of a genuine mistake in the student's solution)
B: To some extent (the response demonstrates some awareness of the exact mistake, but is vague, unclear, or easy to misunderstand)
C: No (the response does not provide any details related to the mistake)
"""
DIMENSIONS = "Mistake_Identification"
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
            data_item["input"] = PROMPT_TEMPLATE.format(definition=DEFINITION, rubric=RUBRIC, history=history, tutor_response=response)
            data_item["output"] = annotation
            data_processed.append(data_item)
    return data_processed


def main():
    # 设置文件路径
    path_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path_input = os.path.join(path_base, "data", "mrbench_v3_devset_dev.json")
    path_output = os.path.join(path_base, "src", "SFT", "LLaMA-Factory", "data", "mrbench_v3_devset_dev_track2.json")
    # 读取数据文件
    data_raw = load_data(path_input)
    # 处理数据
    data_processed = process_data(data_raw)
    # 保存数据
    with open(path_output, "w") as f:
        json.dump(data_processed, f, indent=4)


if __name__ == "__main__":
    main()




