## AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling
### [Paper](https://arxiv.org/abs/2502.15676) | [Project Page](https://chuanyangjin.com/AutoToM) | [Tweet](https://x.com/chuanyang_jin/status/1894737913499246665)

AutoToM is an automated agent modeling method for scalable, robust, and interpretable mental inference. It achieves SOTA on five benchmarks, produces human-like confidence estimates, and supports embodied decision-making. 

![intro](visuals/intro.png)

## Example Usage

*To run AutoToM on MMToM-QA, with the default settings of reduced hypotheses and backwards inference*: 

    python ProbSolver.py --automated --dataset_name "MMToM-QA"

*To run AutoToM on ToMi-1st with a specified model input*: 

    python ProbSolver.py --dataset_name "ToMi-1st" --assigned_model "['State', 'Observation', 'Belief']"

## Requirements

- Install relevant packages:
    - run
    ``
        pip install -r requirements.txt
    ``
- Set your `OPENAI_API_KEY`:

    - On macOS and Linux:
    `export OPENAI_API_KEY='your-api-key'`
    
    - On Windows: `set OPENAI_API_KEY='your-api-key'`

## Experiment 1: Evaluation on ToM Benchmarks

*To run AutoToM on MMToM-QA, with the default settings of reduced hypotheses and backwards inference*: 
    
    cd model
    python ProbSolver.py --automated --dataset_name "MMToM-QA"

## Experiment 2: Evaluation on Classic Cognitive Studies

*To evaluate AutoToM on the cognitive experiments (Food truck scenarios (Desire and belief inference) / Online goal inference)*:
    
    cd experiment_2
    cd food_truck_scenarios # or, cd online_goal_inference
    python eval_AutoToM.py

The final results will be printed at the end of the evaluation.

The analysis code is in `analysis.ipynb` under the folder corresponding to each task.


## Testing AutoToM with customized questions

Please check out ``playground.ipynb``. Simply replace the story and choices with your customized input to see how *AutoToM* discovers agent models and conducts mental inferences!
