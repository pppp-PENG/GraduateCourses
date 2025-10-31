import json
import argparse

prompt = """
你现在是“药学专家智能体”，你的任务是解决复杂的医学问题。
你可以按需调用以下虚拟模块来辅助决策：
    药物数据库查询：获取药物的药理作用、适应症、禁忌症和不良反应。
    临床指南检索：参考最新的权威治疗指南。
    药物相互作用检查器：分析多药联用的风险。
    患者因素评估：综合考虑年龄、肝肾功能、过敏史等。

请按照以下步骤工作：
1. 理解与规划：清晰复述问题，并制定你的解决计划。
2. 执行分析：逐步调用你认为必要的虚拟模块进行分析。
3. 综合判断：汇总各步骤得到的信息，进行综合推理。
4. 给出答案：输出最终答案

开始任务：
问题：
问题类型: {question_type}
问题: {question}
选项: {option}
请直接输出: 正确答案: [选项字母]
"""


def generate_query(data):
    chatgpt_query = prompt
    question = data["question"]
    option = "\n".join([k + ". " + v for k, v in data["option"].items() if v != ""])
    chatgpt_query = chatgpt_query.format_map(
        {"question": question, "option": option, "question_type": data["question_type"]}
    )
    return chatgpt_query


def Prepare_data(args):
    data = []
    # 读取上传的JSON文件
    with open(args.input_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"len:{len(data)}")
    # 根据要求转换
    jsonl_data = []

    for id, item in enumerate(data):
        jsonl_data.append(
            {
                "id": id,
                "query": generate_query(item),
                "model_answer": "",
                "question_type": item["question_type"],
                "groundtruth": item["answer"],
            }
        )

    # 将转换后的数据保存为JSONL文件
    with open(args.output_path, "w", encoding="utf-8") as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Prepare finished, output to '{args.output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for OpenAIGPT generation"
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the output JSONL file."
    )
    args = parser.parse_args()
    Prepare_data(args)
