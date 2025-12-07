import os
import json
from tqdm import tqdm
from openai import OpenAI

# 设置API
BASE_URL = "https://api.nuwaapi.com/v1"
API_KEY = "sk-7NxvusEjS27jAhnl5lSXxIcLxwxdAUatWAuRf4HPeC2Sjj7z"
EVALUATION_DIMENSIONS = ['providing_guidance']

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
                prompt = f'''# Task Description: The assessment of the ###Tutor Response should be based on the following: ###Previous Conversation between Tutor and Student, ###Definitions of criteria and\n\n# Scoring Rubric.\n(1). Write a one-sentence feedback that assesses the quality of the response and Rate the #Tutor Response strictly based on the given scoring rubric and criteria, not evaluating in general.\n(2). After writing feedback, write a score that is an integer between 3 and 1. You should refer to the scoring rubric.\n(3). The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 3 and 1)"\n(4). Please do not generate other opening, closing, or explanations.\n\n# Previous Conversation between Tutor and Student: {history}\n\n# Definitions of criteria: {definition}\n\n# Scoring Rubric: {rubric}\n\n# Tutor Response: {tutor_response}'''
                completion = client.chat.completions.create(
                    model="gemini-2.5-pro-nothinking", #"gemini-2.5-pro-exp-03-25"
                    messages=[
                        {"role": "system", "content": "You are a critic evaluating a tutor's interaction with a student, responsible for providing a clear and objective single evaluation score based on specific criteria. Each assessment must accurately reflect the absolute performance standards."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "# Generate Assessment Score:"},
                    ],
                    temperature=0.3
                )
                response = completion.choices[0].message.content
                item['tutor_responses'][model]['annotation_pred'][evaluation_dimension] = {"response": response}
        save_jsonl(item, path_output_step) # 保存数据
        data_eval.append(item)
    return data_eval

# 保存数据
def save_jsonl(data, path_output_step):
    if not os.path.exists(path_output_step):
        os.makedirs(os.path.dirname(path_output_step), exist_ok=True)
    with open(path_output_step, 'a', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write(',\n')

# 保存数据
def save_json(data, file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 主函数
def main():
    # 设置文件路径
    path_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path_input = os.path.join(path_base, 'data', 'mrbench_v3_devset_dev.json')
    path_output_step = os.path.join(path_base, 'res', 'ICL', 'Track3ProvidingGuidance', 'Gemini-2.5.jsonl')
    path_output = os.path.join(path_base, 'res', 'ICL', 'Track3ProvidingGuidance', 'Gemini-2.5.json')
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
        "mistake_identification": "Has the tutor identified a mistake in a student's response?",
        "mistake_location": "Does the tutor's response accurately point to a genuine mistake and its location?",
        "providing_guidance": "Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint, examples, and so on?",
        "actionability": "Is it clear from the tutor's feedback what the student should do next?",
        }

    mistake_identification_rubric = """
    [Has the tutor identified a mistake in a student's response?]
    Score 3: Yes (The tutor's response recognizes there is an mistake, or believing that further calculations are still needed.)
    Score 2: To some extent (The tutor’s response is vague or ambiguous, without clearly addressing whether an mistake exists.)
    Score 1: No (The tutor's response considered that the question had been completely resolved, or no connection.)
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
    [Is it clear from the tutor's feedback what the student should do next?]
    Score 3: Yes (the tutor's response clearly indicates what the student should do next)
    Score 2: To some extent (the response demonstrates some awareness of what the student should do next, but is vague, unclear, or easy to misunderstand)
    Score 1: No (the response does not provide any details related to what the student should do next)
    """.strip()

    rubric_dict = {
        "mistake_identification": mistake_identification_rubric,
        "mistake_location": mistake_location_rubric,
        "providing_guidance": providing_guidance_rubric,
        "actionability": actionability_rubric
    }
    
    main()