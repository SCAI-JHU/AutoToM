## AutoToM: Scaling Model-based Mental Inference<br> via Automated Agent Modeling

We propose AutoToM, an automated agent modeling method for scalable, robust, and interpretable mental inference.

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
    
    cd Experiment\ 2
    cd Food\ Truck\ Scenarios # or, cd Online\ Goal\ Inference
    python eval_AutoToM.py

The analysis code is in `analysis.ipynb` under the folder corresponding to each task.

## Experiment 3: Embodied Assistance



## Testing AutoToM with customized questions

Please check out ``playground.ipynb``. Simply replace the story and choices with your customized input to see how *AutoToM* discover Bayesian models and conduct inverse planning!
