export LOGPROBS_ENDPOINT="openrouter"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

export LOGPROBS_MODEL="google/gemini-2.5-flash"
export BACKEND_MODEL="google/gemini-2.5-flash"
export RESULTS_DIR="../results-gemini"

python ProbSolver.py --dataset_name=MMToM-QA --seed=0 --automated
