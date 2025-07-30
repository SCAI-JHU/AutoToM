export LOGPROBS_ENDPOINT="openrouter"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

export LOGPROBS_MODEL="qwen/qwen3-235b-a22b-2507"
export BACKEND_MODEL="qwen/qwen3-235b-a22b-2507"
export RESULTS_DIR="../results-qwen"

python ProbSolver.py --dataset_name=MMToM-QA --seed=0 --automated
