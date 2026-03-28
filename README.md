## AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling
### [Paper](https://arxiv.org/abs/2502.15676) | [Project Page](https://chuanyangjin.com/AutoToM) | [Video and Poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/116018) | [Tweet](https://x.com/chuanyang_jin/status/1894737913499246665)

AutoToM is an automated agent modeling method for scalable, robust, and interpretable mental inference. It achieves SOTA on five benchmarks, produces human-like confidence estimates, and supports embodied decision-making.

![intro](visuals/intro.png)

## Example Usage

*To run AutoToM on MMToM-QA, with the default settings of reduced hypotheses and backwards inference*:

```sh
python ProbSolver.py --automated --dataset_name "MMToM-QA"
```

*To run AutoToM on ToMi-1st with a specified model input*:

```sh
python ProbSolver.py --dataset_name "ToMi-1st" --assigned_model "['State', 'Observation', 'Belief']"
```

## Requirements

- Install relevant packages: `pip install -r requirements.txt`
- Set your `OPENAI_API_KEY`:
    - On macOS and Linux: `export OPENAI_API_KEY='your-api-key'`
    - On Windows: `set OPENAI_API_KEY='your-api-key'`

## Experiment 1: Evaluation on ToM Benchmarks

*To run AutoToM on MMToM-QA, with the default settings of reduced hypotheses and backwards inference*:

```sh
cd model
python ProbSolver.py --automated --dataset_name "MMToM-QA"
```

## Experiment 2: Evaluation on Classic Cognitive Studies

*To evaluate AutoToM on the cognitive experiments (Food truck scenarios (Desire and belief inference) / Online goal inference)*:

```sh
cd experiment_2
cd food_truck_scenarios # or, cd online_goal_inference
python eval_AutoToM.py
```

The final results will be printed at the end of the evaluation.

The analysis code is in `analysis.ipynb` under the folder corresponding to each task.


## Experiment 3: Evaluation on Embodied Assistance

*To evaluate AutoToM on the embodied assistance task (Online Watch-And-Help)*:

```sh
git clone -b AutoToM https://github.com/ShunchiZhang/online_watch_and_help
```

Then follow [README](https://github.com/ShunchiZhang/online_watch_and_help/blob/AutoToM/README.md) in the cloned repo for setup and usage.

## Testing AutoToM with customized questions

Please check out ``playground.ipynb``. Simply replace the story and choices with your customized input to see how *AutoToM* discover Bayesian models and conduct inverse planning!

## Citation

Please cite the paper and star this repo if you find it useful, thanks!

```bibtex
@inproceedings{zhang2025autotom,
  title={AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling},
  author={Zhining Zhang and Chuanyang Jin and Mung Yao Jia and Shunchi Zhang and Tianmin Shu},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=oeZZusZheP}
}
```
