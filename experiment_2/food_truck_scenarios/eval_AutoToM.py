import os
import sys

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model")))
def get_full_name(d):
    if d == "K":
        return "Korean"
    if d == "L":
        return "Lebanese"
    if d == "M":
        return "Mexican"

def translate_truck(d, sp):
    if d[0] == "N":
        return f"No truck is in spot {sp}"
    return f"The {get_full_name(d[0])} truck is in spot {sp}, {d[1]} line of sight"

def see_truck(d):
    if d[0] == "N":
        return "sees no truck in spot B"
    
    return f"sees the {get_full_name(d[0])} truck in spot B"

def get_vis(d):
    if d == 1:
        return "within"
    else:
        return "without"

data = {
    # last three: change of visibility, second action, goes to A(1)/B(0)
    "1": ("K", 1, "L", 0, 1, 1, 1),
    

    "2": ("K", 1, "L", 0, 1, 1, 0),


    "5": ("K", 1, "L", 0, 0, 0, 0),


    "16": ("K", 1, "N", 0, 1, 1, 1),


    "18": ("K", 1, "N", 0, 0, 0, 0),


    "55": ("K", 1, "L", 0, 1, 0, 0),


    "56": ("K", 1, "N", 0, 1, 0, 0)
}

def get_story(d):

    intro = "Andy is having lunch from one of the three food trucks: Korean, Lebanese or Mexican."
    spot_A_truck = [d[0], get_vis(d[1])]
    spot_B_truck = [d[2], get_vis(d[3])]
    state_desc = f"{translate_truck(spot_A_truck, 'A')}. {translate_truck(spot_B_truck, 'B')}."

    is_change_visibility = bool(d[4])
    is_second_action = bool(d[5])
    goes_to_A = bool(d[6])

    if is_change_visibility:
        first_action = f"Andy walks past the {get_full_name(spot_A_truck[0])} truck to see what is in spot B. He then {see_truck(spot_B_truck)}."
    else:
        first_action = f"Andy walks to the {get_full_name(spot_A_truck[0])} truck."

    if is_second_action:
        if goes_to_A:
            second_action = f"Then, instead of going to spot B, Andy walks to the {get_full_name(spot_A_truck[0])} truck in spot A."
        else:
            second_action = f"Then, Andy walks to the {get_full_name(spot_B_truck[0])} truck in spot B."
    else:
        second_action=""

    story = f"""{intro}
{state_desc}
{first_action} {second_action}"""
    return story

autotom_final_results = {}

for k, v in data.items():
    d = v
    story = get_story(d)
    print('episode', k, story)
    
    question = "What is Andy's target truck?"
    choices = ["Andy wants to have lunch from the Lebanese truck.", 
            "Andy wants to have lunch from the Mexican truck.",
            "Andy wants to have lunch from the Korean truck."]
    belief_choices = [
        "Andy believes that there is no truck in spot B, and the Mexican truck is not around.",
        "Andy believes that the Mexican truck is in spot B, and the Lebanese truck is not around.",
        "Andy believes that the Lebanese truck is in spot B, and the Mexican truck is not around.",
    ]
    dataset_name = "food_truck"
    episode_name = dataset_name + k
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
            predefined_belief_hypotheses=belief_choices,
            use_all_timesteps=True,
            rational_agent_statement=True,
            approximate=True
        )
    import TimestepInference

    final_probs, model_record = solver.solve()
    init_belief_probs = TimestepInference.init_belief_probs
    autotom_final_results[k] = (final_probs, init_belief_probs)
    print('episode' + k, 'RESULTS:\n', 'goal_probs =', final_probs, '\ninit_belief_probs =', init_belief_probs)

print("AutoToM final results", autotom_final_results)
