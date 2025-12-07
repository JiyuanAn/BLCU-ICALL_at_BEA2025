import os
import json
from tqdm import tqdm
from openai import OpenAI

# 设置API
BASE_URL = "http://202.112.194.80:8880/v1"
API_KEY = "EMPTY"
EVALUATION_DIMENSIONS = ['actionability'] #['mistake_identification', 'mistake_location', 'providing_guidance', 'actionability']

PROMPT_TEMPLATE = """Is it clear from the tutor's latest reply what the student should do next?
- A: Yes (the response provides clear suggestions on what the student should do next)
- B: To some extent (the response indicates that something needs to be done, but it is not clear what exactly that is)
- C: No (the response does not suggest any action on the part of the student (e.g., it simply reveals the final answer))

* Conversation History: “{history}”

* Teacher's Latest Reply: “{tutor_response}”
"""

# 加载数据
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_deepseek_response(data, client, path_output_step):
    data_eval = []
    # 数据准备
    for item in tqdm(data, desc="Processing conversations ", position=0, leave=True):
        history = item['conversation_history']
        for model in tqdm(item['tutor_responses'], desc="Evaluating models        ", position=1, leave=False):
            tutor_response = item['tutor_responses'][model]['response']
            if 'annotation_pred' not in item['tutor_responses'][model]:
                item['tutor_responses'][model]['annotation_pred'] = {}
            for evaluation_dimension in tqdm(EVALUATION_DIMENSIONS, desc=f"Evaluating dimensions    ", position=2, leave=False):
                definition = definition_list[evaluation_dimension]
                rubric = rubric_dict[evaluation_dimension]
                prompt = PROMPT_TEMPLATE.format(history=history, tutor_response=tutor_response)
                completion = client.chat.completions.create(
                    model="Qwen-14B_T4", #gemini-2.5-pro-preview-03-25
                    messages=[
                        {"role": "system", "content": "You are a Senior Teaching Supervisor."},
                        {"role": "user", "content": prompt}
                    ],
                    # temperature=0.3
                )
                while True:
                    try:
                        response = completion.choices[0].message.content
                        break
                    except:
                        continue
                # response = response.split('</think>')[1].strip()
                item['tutor_responses'][model]['annotation_pred'][evaluation_dimension] = {"response": response}
        save_jsonl(item, path_output_step) # 保存数据
        data_eval.append(item)
    return data_eval

# 保存数据
def save_jsonl(data, path_output_step):
    with open(path_output_step, 'a', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write(',\n')

# 保存数据
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



# 主函数
def main():
    # 设置文件路径
    path_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(path_base)
    path_input = os.path.join(path_base, 'data', 'mrbench_v3_testset.json')
    path_output_step = os.path.join(path_base, 'res', 'Qwen-14B-V0201_2.jsonl')
    path_output = os.path.join(path_base, 'res', 'Qwen-14B-V0201_02.json')
    # 初始化OpenAI客户端
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    # 读取原始文件
    data = load_json(path_input)
    # 获取导师响应
    data_new = get_deepseek_response(data, client, path_output_step)
    # 保存至新文件
    save_json(data_new, path_output)


if __name__ == '__main__':
    definition_list = {
        "mistake_identification": "Has the tutor explicitly pointed out that there was a mistake in a student's response?",
        "mistake_location": "Does the tutor's response accurately point to a genuine mistake and its location?",
        "providing_guidance": "Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint, examples, and so on?",
        "actionability": "Is it clear from the tutor's feedback what the student should do next?",
        }

    mistake_identification_rubric = """
    A: Yes (The tutor's response recognizes there is a mistake, or provides some practical guidance.)
    B: To some extent
    C: No (The tutor's response believes that the question had been completely resolved, or no connection.)
    """.strip()

    mistake_location_rubric = """
    [Does the tutor's response accurately point to a genuine mistake and its location?]
    Score 3: Yes (the tutor clearly points to the exact location of a genuine mistake in the student's solution)
    Score 2: To some extent (the response demonstrates some awareness of the exact mistake, but is vague, unclear, or easy to misunderstand)
    Score 1: No (the response does not provide any details related to the mistake)
    """.strip()

    providing_guidance_rubric = """
    [Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint, examples, and so on?]
    Score 3: Yes (the tutor provides guidance that is correct and relevant to the student's mistake)
    Score 2: To some extent (guidance is provided but it is fully or partially incorrect, incomplete, or somewhat misleading)
    Score 1: No (the tutor's response does not include any guidance, or the guidance provided is irrelevant to the question or factually incorrect)
    """.strip()

    actionability_rubric = """
    A: Yes (the response provides clear suggestions on what the student should do next)
    B: To some extent (the response indicates that something needs to be done, but it is not clear what exactly that is)
    C: No (the response does not suggest any action on the part of the student (e.g., it simply reveals the final answer))
    """.strip()

    rubric_dict = {
        "mistake_identification": mistake_identification_rubric,
        "mistake_location": mistake_location_rubric,
        "providing_guidance": providing_guidance_rubric,
        "actionability": actionability_rubric
    }
    
    main()