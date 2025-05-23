
# python human_eval/evaluate_functional_correctness.py data/example_samples.jsonl --problem_file=data/example_problem.jsonl


# python human_eval/evaluate_functional_correctness.py output/glm_mad_5a_5r_complete/samples_round_0.jsonl

expname="mad_8a_3r_WS5"

# python human_eval/evaluate_functional_correctness.py output/$expname/samples.jsonl

python human_eval/evaluate_functional_correctness.py output/$expname/samples_round_1.jsonl
python human_eval/evaluate_functional_correctness.py output/$expname/samples_round_2.jsonl
# python human_eval/evaluate_functional_correctness.py output/$expname/samples_round_3.jsonl
# python human_eval/evaluate_functional_correctness.py output/$expname/samples_round_4.jsonl


