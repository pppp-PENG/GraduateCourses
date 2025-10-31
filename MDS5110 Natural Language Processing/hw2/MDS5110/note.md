```shell
python 1.prepare_data.py --input_path "./data/1.exam.json" --output_path "./data/2.exam_prepared.jsonl"

python "./langchain_datagen_multithread.py" --keys_path "./gpt3keys.txt" --input_path "./data/2.exam_prepared.jsonl" --output_path "./data/3.exam_aftgpt.jsonl" --max_workers 10 --base_url "https://apix.ai-gaochao.cn/v1"

python 3.scorer.py --input_path "./data/3.exam_aftgpt.jsonl" --wrong_ans_path "./data/4.wrong_ans.json" --score_path "./data/4.score.json"
```

Base:
```python
prompt ="""
Question Type: {question_type}
Question: {question}
Options: {option}

请直接输出: 正确答案: [option letters]
"""
[最佳选择题]准确率：0.714  题目总数：35
[配伍选择题]准确率：0.867  题目总数：45
[综合分析选择题]准确率：0.583  题目总数：12
[多项选择题]准确率：0.625  题目总数：8
总分：76  / 满分：100
```

Few Shot:
```python
prompt ="""
你是一名临床药师，需要基于药学知识逐步分析问题。

示例：
问题：患者服用华法林时，应避免合用以下哪种药物？
选项：A) 对乙酰氨基酚 B) 维生素K C) 阿司匹林 D) 奥美拉唑
分析：
- 华法林是抗凝药，通过抑制维生素K依赖的凝血因子合成起作用
- 阿司匹林具有抗血小板作用，会协同增加出血风险
- 维生素K是华法林的拮抗剂，其他选项风险较低
正确答案：C

问题：二甲双胍的禁忌证包括？
选项：A) 1型糖尿病 B) 肾功能不全 C) 妊娠期糖尿病
分析：
- A错误：1型糖尿病可联用胰岛素
- B正确：eGFR<30禁用（反指证）
- C错误：妊娠期可谨慎使用
正确答案：B

Question Type: {question_type}
Question: {question}
Options: {option}

请直接输出: 正确答案: [option letters]
"""
[最佳选择题]准确率：0.743  题目总数：35
[配伍选择题]准确率：0.822  题目总数：45
[综合分析选择题]准确率：0.750  题目总数：12
[多项选择题]准确率：0.625  题目总数：8
总分：77  / 满分：100
```


```
[最佳选择题]准确率：0.800  题目总数：35
[配伍选择题]准确率：0.867  题目总数：45
[综合分析选择题]准确率：0.500  题目总数：12
[多项选择题]准确率：0.625  题目总数：8
总分：78  / 满分：100
```