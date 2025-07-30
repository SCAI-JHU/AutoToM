export LOGPROBS_ENDPOINT="openrouter"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

export LOGPROBS_MODEL="deepseek/deepseek-chat-v3-0324"
export BACKEND_MODEL="deepseek/deepseek-chat-v3-0324"
export RESULTS_DIR="../results-deepseek"

python ProbSolver.py --dataset_name=MMToM-QA --seed=0 --automated
