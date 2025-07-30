conda-activate
tmux new -t llm
tmux ls
tmux a -t llm
conda-deactivate && uv-activate r1_vlm

cd model
bash ../scripts/run-deepseek.sh # skip 120
bash ../scripts/run-gemini.sh
bash ../scripts/run-qwen.sh

ls results-{deepseek,gemini,qwen}/metrics/ | wc -l

watch -n 2 'for dir in deepseek gemini qwen; do echo "$dir"; ls "results-$dir/metrics/" | wc -l; done'
