import re
import time
from tqdm import tqdm
import copy
import networkx as nx
import numpy as np
import json
import os
import random

with open("./problems_100.json", 'r') as f:
    problems = json.load(f)
print(len(problems))

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

import time
start_time = time.time()

################################# 设置使用的模型和实验路径名称
num_rounds = 3
llm_configs = [glm_config, gpt35_config, internlm20b_config, qwen7b_config, gemma27b_config, gpt4omini_config, qwen32b_config, qwen72b_config] 
adj_matrix = np.array([[0, 1, 1, 0, 0, 0, 0, 1], [1, 0, 1, 1, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1, 1, 0]])
expname = "mad_8a_3r_WS2" 
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

# # deg=2的环图
# G = nx.cycle_graph(num_agents)
# adj_matrix = np.array(nx.adjacency_matrix(G).todense())  # 获取邻接矩阵并转化为密集矩阵
# print("图的邻接矩阵：\n", adj_matrix)
# print("节点度数：", dict(G.degree()))

# # 完全图
# G = nx.complete_graph(num_agents)  # 生成一个包含n个节点的完全图
# adj_matrix = np.array(nx.adjacency_matrix(G).todense())  # 获取邻接矩阵并转化为密集矩阵
# print("图的邻接矩阵：\n", adj_matrix)
# print("节点度数：", dict(G.degree()))

# G = nx.star_graph(num_agents-1)  # star_graph(n-1) 会生成一个包含 n 个节点的星形图，n-1是中心节点的邻居数
# adj_matrix = np.array(nx.adjacency_matrix(G).todense())  # 获取邻接矩阵并转化为密集矩阵
# print("图的邻接矩阵：\n", adj_matrix)
# print("节点度数：", dict(G.degree()))
    
# 每个点deg=4的随机图（n>=5）
# def generate_graph(n):
#     # 创建一个空图
#     G = nx.Graph()

#     # 添加n个节点
#     G.add_nodes_from(range(n))

#     # 如果n小于5，无法构造每个节点度为4的无向图，返回空图
#     if n < 5:
#         print("无法构造每个节点度为4的图，节点数应至少为5。")
#         return None

#     # 创建一个每个节点度为4的图
#     # 使用 NetworkX 的 Barabási-Albert 模型生成具有给定度数的无向图
#     random.seed(20)
#     G = nx.random_regular_graph(4, n)  # 生成每个节点度为4的随机图

#     # 返回生成的图
#     return G


# 生成图
# G = generate_graph(num_agents)
# if G:
#     adj_matrix = np.array(nx.adjacency_matrix(G).todense())  # 获取邻接矩阵并转化为密集矩阵
#     print("图的邻接矩阵：\n", adj_matrix)
#     print("节点度数：", dict(G.degree()))


print("图的邻接矩阵：\n", adj_matrix)

# 构建agents
system_message = "You are an expert skilled in playing chess."
prompt_template = """Given the chess game "{game}", give one valid destination square for the chess piece at "{piece}". Give a one line explanation of why your destination square is a valid move. State your final answer in a new line with a 2 letter response following the regex [a-h][1-8].
""".strip()

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
accumulated_usage = np.zeros((len(problems), num_rounds, num_agents, 2))

task_id = 0
for k,v in problems.items():
    print(f"====={k}=====")
    tmp = v['input']
    game, piece = tmp.rsplit(' ', 1)
    question_prompt = prompt_template.format(game=game, piece=piece)

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
                        prompt += """\n\n Using the answer from other agents as additional advice, can you give an updated answer? Give a one line explanation of why your destination square is a valid move. State your final answer in a new line with a 2 letter response following the regex [a-h][1-8]."""
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
    task2messages[k] = messages
    
    task_id += 1
    print("time:", time.time()-start_time)

# 处理得到每个问题、每轮、每个agent的input/output token消耗
accumulated_usage = accumulated_usage.reshape(len(problems)*num_rounds, num_agents, 2)
token_usage = np.zeros(accumulated_usage.shape)
token_usage[0] = accumulated_usage[0]
for i in range(1, len(problems)*num_rounds):
    token_usage[i] = accumulated_usage[i] - accumulated_usage[i-1]
    
accumulated_usage = accumulated_usage.reshape(len(problems), num_rounds, num_agents, 2)
token_usage = token_usage.reshape(len(problems), num_rounds, num_agents, 2)

np.save(os.path.join(filepath, 'token_usage.npy'), token_usage)
with open(os.path.join(filepath, 'task2messages.json'), 'w') as f:
    json.dump(task2messages, f, indent=4)

for agent in agents:
    agent.print_usage_summary()

print("time:", time.time()-start_time)