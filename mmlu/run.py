import re
import time
from tqdm import tqdm
import copy
import numpy as np
import pandas as pd
import json

file_path = 'mmlu_pro1_selected.csv'
data = pd.read_csv(file_path, encoding='latin1')

# 提取前三列到对应的数组
questions = data.iloc[0:101, 0].tolist()  # 第一列存入 questions
choices = data.iloc[0:101, 1].tolist()    # 第二列存入 choices
answers = data.iloc[0:101, 2].tolist()    # 第三列存入 answers

num = len(questions)

from autogen import ConversableAgent
# silicon flow
siliconflow_key = 'Put your siliconflow key here'

temperature = 0.0

qwen7b_config = {"model": "Qwen/Qwen2.5-7B-Instruct", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "temperature": temperature,
            "cache_seed": None,
            "price" : [0.0, 0.0]}
qwen32b_config = {"model": "Qwen/Qwen2.5-32B-Instruct", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "temperature": temperature,
            "cache_seed": None,
            "price" : [0.0, 0.0]}
qwen72b_config = {"model": "Qwen/Qwen2.5-72B-Instruct", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "temperature": temperature,
            "cache_seed": None,
            "price" : [0.0, 0.0]}

glm_config = {"model": "THUDM/glm-4-9b-chat", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "temperature": temperature,
            "cache_seed": None,
            "price" : [0.0, 0.0]}
deepseek_config = {"model": "deepseek-ai/DeepSeek-V2.5", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "temperature": temperature,
            "cache_seed": None,
            "price" : [0.0, 0.0]}
llama8b_config = {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "temperature": temperature,
            "cache_seed": None,
            "price" : [0.0, 0.0]}
llama70b_config = {"model": "meta-llama/Meta-Llama-3.1-70B-Instruct", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "temperature": temperature,
            "cache_seed": None,
            "price" : [0.0, 0.0]}

internlm20b_config = {"model": "internlm/internlm2_5-20b-chat", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "temperature": temperature,
            "cache_seed": None,
            "price" : [0.0, 0.0]}
yi34b_config = {"model": "01-ai/Yi-1.5-34B-Chat-16K", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "temperature": temperature,
            "cache_seed": None,
            "price" : [0.0, 0.0]}
gemma27b_config = {"model": "google/gemma-2-27b-it", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "temperature": temperature,
            "cache_seed": None,
            "price" : [0.0, 0.0]}
openai_key = "Put your openai key here"
gpt35_config = {"model": "gpt-35-turbo",
            "api_type": "azure",
            "api_key": openai_key,
            "temperature": temperature,
            "cache_seed": None,
            "base_url": "Put your openai base url here",
            "api_version": "Put your openai api version here"}
gpt4omini_config = {"model": "gpt-4o-mini",
            "api_type": "azure",
            "api_key": openai_key,
            "temperature": temperature,
            "cache_seed": None,
            "base_url": "Put your openai base url here",
            "api_version": "Put your openai api version here"}


import os
################################# 设置使用的模型和实验路径名称

num_rounds = 3
llm_configs = [glm_config, gpt35_config, internlm20b_config, qwen7b_config, gemma27b_config, gpt4omini_config, qwen32b_config, qwen72b_config] 
adj_matrix = np.array([[0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1, 1, 0], [0, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0]])
expname = "mad_8a_3r_WS1" 
#################################

num_agents = len(llm_configs)
print('#Agents:', num_agents)
print('#Rounds:', num_rounds)

# multi agent时temperature设为1
temperature = 1.0
for c in llm_configs:
    c['temperature'] = temperature

filepath = f"./output/{expname}/"
if not os.path.exists(filepath):
    os.makedirs(filepath)
    print('make new dir',filepath)

# 构建agents
system_message = f""" 
You are an expert answering several multiple-choice questions. You will be given the question directly,and a [] and several choices in it.Please name those choices as A~Z in the order of the choices.For example, if you're given ['1','2','3'],then please use A to represent '1',B to represent '2' and C to represent '3' .Explain your answer, and please strictly obey the rule that the last sentence of your response should be "the answer is " plus a signle letter.Do not put a dot at the end.
"""
agents = []
for i, config in enumerate(llm_configs):
    agent = ConversableAgent(
                name=f'agent_{i}',
                system_message=system_message,
                llm_config={"config_list": [config]},
                code_execution_config=False,
                function_map=None,
                max_consecutive_auto_reply=1000000,
                human_input_mode="NEVER",
            )
    agents.append(agent)
    
human_proxy = ConversableAgent(
    "human_proxy",
    llm_config=False,
    human_input_mode="ALWAYS",
)

task2messages = {}
accumulated_usage = np.zeros((len(questions), num_rounds, num_agents, 2))

task_id = 0
for nn in range(0,num):
    question_exam = questions[nn]
    choice = choices[nn]
    question_prompt = f"""
    Can you answer the following question as accurately as possible? {question_exam}: choices are in the [] below:{choice}. Please name those choices as A~Z in the order of the choices.Always remember that you can only give ana answer that exists.For example, if you're given ['1','2','3'],then please use A to represent '1',B to represent '2' and C to represent '3' .In this example,don't give answers besides A,B and C. The number of choices can vary from 4 to more.Explain your answer, and please strictly obey the rule that the last sentence of your response should be "the answer is " plus a signle letter.Do not put a dot at the end.
    """.strip()

    for _ in range(10):
        try:

            # 每道题开始时，清除agents历史记录
            human_proxy.clear_history()
            for i, agent in enumerate(agents):
                agent.clear_history()
            
            for round_id in range(num_rounds):
                print(f'Round {round_id}')
                for agent_id, agent in enumerate(agents):
                    if round_id == 0: # 第一轮：每个agent分别回答
                        prompt = question_prompt
                    else: # 第2-n轮：每个agent根据相邻agent的答案，更新回答
                        prompt = "These are the solutions to the problem from other agents: "
                        for j in range(num_agents):
                            if adj_matrix[agent_id, j] == 1:
                                agent_response = agents[j].last_message(human_proxy)['content']
                                prompt += f"\n\n One agent solution: ```{agent_response}```"
                        prompt += """\n\n Using the answer from other agents as additional advice, can you give an updated answer? Please name those choices as A~Z in the order of the choices.Always remember that you can only give ana answer that exists.For example, if you're given ['1','2','3'],then please use A to represent '1',B to represent '2' and C to represent '3' .In this example,don't give answers besides A,B and C. The number of choices can vary from 4 to more.Explain your answer, and please strictly obey the rule that the last sentence of your response should be "the answer is " plus a signle letter.Do not put a dot at the end."""
                    human_proxy.send(message=prompt,
                            recipient=agent,
                            request_reply=True,
                            silent=True
                            )
                    # 保存token用量
                    current_usage = agent.get_total_usage()
                    current_usage = list(current_usage.values())[1]
                    accumulated_usage[task_id, round_id, agent_id, 0] = current_usage['prompt_tokens']
                    accumulated_usage[task_id, round_id, agent_id, 1] = current_usage['completion_tokens']
                    time.sleep(0.5)
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)
            
    # get messages
    messages = dict(human_proxy.chat_messages)
    messages = list(messages.values())
    task2messages[nn] = messages
    
    task_id += 1


accumulated_usage = accumulated_usage.reshape(len(questions)*num_rounds, num_agents, 2)
token_usage = np.zeros(accumulated_usage.shape)
token_usage[0] = accumulated_usage[0]
for i in range(1, len(questions)*num_rounds):
    token_usage[i] = accumulated_usage[i] - accumulated_usage[i-1]
    
accumulated_usage = accumulated_usage.reshape(len(questions), num_rounds, num_agents, 2)
token_usage = token_usage.reshape(len(questions), num_rounds, num_agents, 2)

# 保存token消耗和agent的原始回答
np.save(os.path.join(filepath, 'token_usage.npy'), token_usage)
with open(os.path.join(filepath, 'task2messages.json'), 'w') as f:
    json.dump(task2messages, f, indent=4)


correct_2r = 0
correct_3r = 0

# 评价
for nn in range(0,num):
    nnn = str(nn)
    num_correct_2r = 0
    num_correct_3r = 0
    for na in range(0,num_agents):
        answer_2r = task2messages[nnn][na][3]["content"]
        answer_3r = task2messages[nnn][na][5]["content"]
        if (len(answer_2r) >0 and answer_2r[-1] == answers[nn]) or (len(answer_2r) > 1 and answer_2r[-2] == answers[nn] ):
            num_correct_2r = num_correct_2r + 1
        if (len(answer_3r) >0 and answer_3r[-1] == answers[nn]) or (len(answer_3r) > 1 and answer_3r[-2] == answers[nn] ):
            num_correct_3r = num_correct_3r + 1
    if num_correct_2r > num_agents/2:
        correct_2r = correct_2r+1
    if num_correct_3r > num_agents/2:
        correct_3r = correct_3r+1


acc_2r = correct_2r/num
acc_3r = correct_3r/num

print(acc_2r)
print(acc_3r)






