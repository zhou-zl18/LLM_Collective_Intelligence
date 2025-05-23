import os
import json
import time
import networkx as nx
import numpy as np
from util import last_boxed_only_string
from math_equivalence import is_equiv
from autogen import ConversableAgent
filepath = "./ER5_1.txt"
temperature = 1.0
num_rounds=3

siliconflow_key = 'Put your siliconflow key here'
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

llm_configs = [glm_config, gpt35_config, internlm20b_config, qwen7b_config, gemma27b_config, gpt4omini_config, qwen32b_config, qwen72b_config] 
adj_matrix = np.array([[0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]])
num_agents = len(llm_configs)

train_prompt = "Simplify your answer as much as possible. Put your final answer inside \\boxed{}. Here are some examples:" + "\n" + "Problem: What is $\\left(\\frac{7}{8}\\right)^3 \\cdot \\left(\\frac{7}{8}\\right)^{-3}$?" + "\n" + "Answer: $\\boxed{1}$"
train_prompt += "\n" + "###" + "\n" + "Problem: In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?" + "\n" +"Answer: $\\boxed{15}$"
train_prompt += "\n" +"###" + "\n" + "Problem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$" + "\n" + "Answer: $\\boxed{\\sqrt{59}}$"
train_prompt += "\n" + "###" + "\n" + "Problem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?" + "\n" + "Answer: $\\boxed{\\frac{1}{32}}$"
train_prompt += "\n" + "###" + "\n" + "Problem: The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?" + "\n" + "Answer: $\\boxed{181}$"
train_prompt += "\n" + "###" + "\n" + "Problem: Calculate $6 \\cdot 8\\frac{1}{3}" + "\n" + "Answer: $\\boxed{50}$"
train_prompt += "\n" + "###" + "\n" + "Problem: When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?" + "\n" + "Answer: $\\boxed{2}$"
train_prompt += "\n" + "###" + "\n" + "Problem: How many zeros are at the end of the product 25 $\\times$ 240?" + "\n" + "Answer: $\\boxed{3}$" + "\n" + "###"

agents = []
order=["first","second","third","fourth","fifth","sixth","seventh","eighth"]
for i, config in enumerate(llm_configs):
        agent = ConversableAgent(
                    name=f'agent_{i}',
                    system_message="You are the "+order[i]+" in a group of three to solve the counting and probability problem."+train_prompt,
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

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


rootdir = "./sample"

def run(max=-1):
    outputs = []
    answers = []
    levels = []
    correct = [0,0,0]
    total = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            with open(os.path.join(subdir, file), 'r') as fp:
                total += 1
                try:
                    problem_data = json.load(fp)
                except Exception as e:
                    print(f"Error loading JSON from {file}", e)
                    raise e
                prob_level = problem_data["level"]
                try:
                    prob_level = int(prob_level.split("Level ")[1])
                except:
                    prob_level = None
                answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))
                print("Correct answer:"+answer)
                with open(filepath, 'a') as f:
                    f.write("Problem {}:\n".format(total)+problem_data["problem"]+"\n")

                output=[]
                flag=0
                for _ in range(10):
                    try:
                        right=[0,0,0]
                        human_proxy.clear_history()
                        for i, agent in enumerate(agents):
                            agent.clear_history()
            
                        for round_id in range(num_rounds):
                            responses=[]
                            print(f'Round {round_id+1}'+':')
                            for agent_id, agent in enumerate(agents):
                                if round_id == 0: # 第一轮：每个agent分别回答
                                    prompt = problem_data["problem"]
                                    prompt+="\nWalk through the final calculation.Put your final answer inside \\boxed{}."
                                else: # 第2-n轮：每个agent根据相邻agent的答案，更新回答
                                    prompt = "These are the solutions to the problem from other agents: "
                                    for j in range(num_agents):
                                        if adj_matrix[agent_id, j] == 1:
                                            agent_response = agents[j].last_message(human_proxy)['content']
                                            prompt += f"\n\n One agent solution: ```{agent_response}```"
                                    prompt += """\n\n Using the reasoning from other agents as additional advice.Examine your solution and that from other agents and give an updated answer.Walk through the final calculation and put your final answer inside \\boxed{}. """
                                if flag==1:
                                    prompt=prompt[:6700]
                                human_proxy.send(message=prompt,recipient=agent,request_reply=True,silent=True)
                                res = agent.last_message(human_proxy)['content']
                                print(f'Agent {agent_id}'+':\n'+res)
                                responses.append(res)
                            model_outputs=[]
                            num=0
                            for result in responses:
                                model_output=remove_boxed(last_boxed_only_string(result))
                                model_outputs.append(model_output)
                                try:
                                    equiv = is_equiv(model_output, answer)
                                except:
                                    equiv = False
                                if equiv:
                                    num += 1
                            if num>=num_agents//2+1:
                                right[round_id]=1
                            output.append(model_outputs)
                            print("Model output:"+str(model_outputs))
                            print("Acc: "+str(correct[round_id]) + "/" + str(total))
                        for k in range(num_rounds):
                            if right[k]==1:
                                correct[k]+=1
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        flag=1
                        with open(filepath, 'a') as f:
                            f.write(f"Error: {e}\n\n")
                        time.sleep(30)

                print("--------------------------------------------")
                levels.append(prob_level)
                outputs.append(output)
                answers.append(answer)
                
            if total>=33:
                break
            if max > 0 and total > max:
                break
        if max > 0 and total > max:
            break
        token_num=0
        for agent in agents:
            current_usage = agent.get_total_usage()
            print(str(current_usage))
            current_usage = list(current_usage.values())[1]
            token_num+=current_usage['total_tokens']

        with open(filepath, 'a') as f:
            f.write("levels:\n"+str(levels)+'\n')
            f.write("llm outputs:\n"+str(outputs)+'\n')
            f.write("correct answer:\n"+str(answers)+'\n')
            f.write("Acc:\n")
            for i in range(num_rounds):
                f.write(f'Round {i+1}')
                f.write(": "+str(correct[i]) + "/" + str(total)+'\n')
            f.write("Total tokens:"+str(token_num)+'\n')

if __name__ == "__main__":
    run()
