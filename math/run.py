import os
import json
import time
from util import last_boxed_only_string
from math_equivalence import is_equiv
from autogen import ConversableAgent
siliconflow_key = 'Put your siliconflow key here'
filepath = "./qwen3.txt"

glm_config = {"model": "THUDM/glm-4-9b-chat", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "price" : [0.0, 0.0]}
qwen7_config = {"model": "Qwen/Qwen2.5-7B-Instruct", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "price" : [0.0, 0.0]}
qwen32_config = {"model": "Qwen/Qwen2.5-32B-Instruct", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "price" : [0.0, 0.0]}
qwen72_config = {"model": "Qwen/Qwen2.5-72B-Instruct", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "price" : [0.0, 0.0]}
deepseek_config = {"model": "deepseek-ai/DeepSeek-V2.5", 
            "api_key": siliconflow_key, 
            "base_url":"https://api.siliconflow.cn/v1",
            "price" : [0.0, 0.0]}
llm_configs = [qwen7_config, qwen32_config, qwen72_config]

def call_engine(train_prompt, problem):
    test_question = "\n" + problem
    agents = []
    order=["first","second","third"]
    responses=[]
    for i, config in enumerate(llm_configs):
        agent = ConversableAgent(
                    name=f'agent_{i}',
                    system_message="You are the "+order[i]+" in a group of three to solve the counting and probability problem."+train_prompt,
                    llm_config={"config_list": [config]},
                    code_execution_config=False,
                    function_map=None,
                    human_input_mode="NEVER",
                )
        agents.append(agent)
    
    human_proxy = ConversableAgent(
                "human_proxy",
                llm_config=False,
                human_input_mode="ALWAYS",
            )
    for i, agent in enumerate(agents):
        human_proxy.send(message=test_question,recipient=agent,request_reply=True)
    for i, agent in enumerate(agents):
        prompt = "These are the solutions to the problem from other agents: "
        for j in range(len(agents)):
            if j != i:
                agent_response = agents[j].last_message(human_proxy)['content']
                prompt += f"\n\n One agent solution: ```{agent_response}```"
        prompt += """\n\n Using the reasoning from other agents as additional advice.Examine your solution and that other agents, can you give an updated answer?  """
        human_proxy.send(message=prompt,recipient=agent,request_reply=True)
        responses.append(agent.last_message(human_proxy)['content'])
    return responses

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

train_prompt = "Simplify your answer as much as possible. Here are some examples:" + "\n" + "Problem: What is $\\left(\\frac{7}{8}\\right)^3 \\cdot \\left(\\frac{7}{8}\\right)^{-3}$?" + "\n" + "Answer: $\\boxed{1}$"
train_prompt += "\n" + "###" + "\n" + "Problem: In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?" + "\n" +"Answer: $\\boxed{15}$"
train_prompt += "\n" +"###" + "\n" + "Problem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$" + "\n" + "Answer: $\\boxed{\\sqrt{59}}$"
train_prompt += "\n" + "###" + "\n" + "Problem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?" + "\n" + "Answer: $\\boxed{\\frac{1}{32}}$"
train_prompt += "\n" + "###" + "\n" + "Problem: The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?" + "\n" + "Answer: $\\boxed{181}$"
train_prompt += "\n" + "###" + "\n" + "Problem: Calculate $6 \\cdot 8\\frac{1}{3}" + "\n" + "Answer: $\\boxed{50}$"
train_prompt += "\n" + "###" + "\n" + "Problem: When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?" + "\n" + "Answer: $\\boxed{2}$"
train_prompt += "\n" + "###" + "\n" + "Problem: How many zeros are at the end of the product 25 $\\times$ 240?" + "\n" + "Answer: $\\boxed{3}$" + "\n" + "###"

rootdir = "./sample"

def run(max=-1):
    begin=time.time()
    outputs = []
    answers = []
    levels = []
    times=[]

    correct = 0
    total = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            with open(os.path.join(subdir, file), 'r') as fp:
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
                start=time.time()
                answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))
                results=call_engine(train_prompt, problem_data["problem"])
                model_outputs=[]
                num=0
                for result in results:
                    model_output=remove_boxed(last_boxed_only_string(result))
                    model_outputs.append(model_output)
                    try:
                        equiv = is_equiv(model_output, answer)
                    except:
                        equiv = False
                    if equiv:
                        num += 1
                if num>=2:
                    correct+=1
                total += 1
                t=time.time()-start
                levels.append(prob_level)
                outputs.append(model_outputs)
                answers.append(answer)
                times.append(t)
                print("Time:")
                print(t)
                print("Model output:")
                print(model_outputs)
                print("Correct answer:")
                print(answer)
                print(str(correct) + "/" + str(total))
                print("--------------------------------------------")
            if total>=50:
                break
            if max > 0 and total > max:
                break
        if max > 0 and total > max:
            break
        totaltime=time.time()-begin
        with open(filepath, 'a') as f:
            f.write("levels:\n"+str(levels)+'\n')
            f.write("llm outputs:\n"+str(outputs)+'\n')
            f.write("correct answer:\n"+str(answers)+'\n')
            f.write("acc:"+str(correct) + "/" + str(total)+'\n')
            f.write("time:"+str(times)+'\n')
            f.write("Total time:"+str(totaltime)+'\n')

if __name__ == "__main__":
    run()
