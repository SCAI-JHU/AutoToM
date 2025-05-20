import numpy as np
from copy import deepcopy
import os
import sys

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model")))
data = {
    "1": """Andy moves closer to all three letters (A, B, and C) where C is closer. 
Andy continues walking.
Andy does not move directly to C but walks towards A and B.
Andy keeps walking.
Andy walks directly towards A, but not B or C.""",
    "2": """Andy moves closer to all letters (A, B, and C) where C is closer.
Andy continues walking.
Andy does not move directly to C but walks towards A and B.
Andy walks directly towards B, but not A or C.
Andy keeps walking.
""",
    "3": """Andy moves closer to all three letters (A, B, and C) where C is closer.
Andy continues walking, but not on the shortest path to B.
Andy does not walk towards C but walks towards A and B.
Andy walks directly towards B, but not A or C.
Andy keeps walking.
""",
    "4": """Andy moves closer to all three letters (A, B, and C) where C is closer.
Andy does not move on the shortest path to C but walks towards A and B, while being closer to B.
Andy keeps walking.
Andy walks directly towards A, but not B or C.
Andy keeps walking.
""",
    "5": """Andy moves closer to all three letters (A, B, and C) while being roughly on the closest path to A.
Andy does not move on the shortest path to A but walks towards B and C.
Andy keeps walking.
Andy walks towards A and C, while their path is not aligned with the direction of B.
Andy walks towards A, but not B or C.
""",
    "6": """Andy moves closer to all three letters (A, B, and C) while being roughly on the closest path to A.
Andy does not move on the shortest path to A but walks towards B and C.
Andy keeps walking.
Andy walks towards B, but not A or C.
Andy keeps walking.
""",
}
goal_likelihoods = {}
accumulated_goal_lh = {}

for k, v in data.items():
    d = v
    story = d
    print('episode', k, story)
    # quit()
    question = "Which one is Andy's target letter?"
    choices = ["Andy's target letter is A.", 
            "Andy's target letter is B.",
            "Andy's target letter is C."]
    dataset_name = "online_goal"
    episode_name = "online_goal" + k
    verbose = False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(script_dir, "../../model"))
    sys.path.append(model_dir)

    from ProbSolver import ProblemSolver, argmax, argmin
    solver = ProblemSolver(
            story=story,
            question=question,
            choices=choices,
            K=1,
            assigned_model=[],
            model_name="automated",
            episode_name=episode_name,
            llm="gpt-4o",
            verbose=verbose,
            dataset_name=dataset_name,
            hypo_method="guided",
            nested=False,
            video_id=None,
            answerfunc=argmax,
            back_inference=True,
            reduce_hypotheses=True,
            precomputed_states=None,
            precomputed_actions=None,
            prev_hyp=None,
            no_model_adjustment=False,
            recursion_depth=None,
            use_all_timesteps=True,
            rational_agent_statement=True
        )
    import TimestepInference
    
    final_probs, model_record = solver.solve()
    print('episode' + k, 'RESULTS:\n', 'goal_probs =', final_probs)

    goal_likelihoods = TimestepInference.all_time_goal_likelihood
    accumulated_goal_lh[k] = []
    accumulated_prob = np.array([1.0, 1.0, 1.0])
    for i in range(5):
        accumulated_prob *= goal_likelihoods[i]
        accumulated_prob /= np.sum(accumulated_prob)
        accumulated_goal_lh[k].append(deepcopy(accumulated_prob))

print(accumulated_goal_lh)
