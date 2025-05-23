from human_eval.data import write_jsonl, read_problems

problems = read_problems()

def generate_one_completion(prompt):
    return f"{prompt} <mask>"

num_samples_per_task = 1
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
print(samples)
write_jsonl("samples.jsonl", samples)