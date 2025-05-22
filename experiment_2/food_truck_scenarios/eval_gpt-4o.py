import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(script_dir, "../../model"))
sys.path.append(model_dir)

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
    return f"the {get_full_name(d[0])} truck is in spot {sp}, {d[1]} Andy's line of sight"

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

from utils import gpt_request, enh_print


def get_story(d):

    intro = "Andy is having lunch from one of the three food trucks: Korean, Lebanese or Mexican."
    spot_A_truck = [d[0], get_vis(d[1])]
    spot_B_truck = [d[2], get_vis(d[3])]
    state_desc = f"Initially, {translate_truck(spot_A_truck, 'A')}, and {translate_truck(spot_B_truck, 'B')}."

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


for k, v in data.items():
    d = v
    story = get_story(d)
    print('episode', k)
    # quit()
    question = f"""Read the story and rate for Andy's desire levels. 
{story}
What are Andy's desire ratings for the three trucks, on a scale of 1 to 7, assuming that Andy is very sure of their belief and will not perform unnecessary actions?
Output in the format of [x, y, z], where x, y, z are the Andy's desire ratings of Lebanese truck, Mexican truck, and Korean truck. Higher the number is, the more Andy wants to eat at the food truck.
Example Output: [1, 4, 7]. Only output the list and do not include any explanations in the response.
Output:"""
    # print(question)
    ans, cost = gpt_request(question)
    enh_print('desire: ' + ans)
    question = f"""Read the story and rate for Andy's initial beliefs. 
{story}
What are Andy's initial beliefs about the truck in spot B, on a scale of 1 to 7, assuming that Andy is very sure of his beliefs and will not perform unnecessary actions?
You should rate the beliefs retrospectively, based on what Andy thought was in spot B at the beginning of the story.
Output in the format of [x, y, z], where x, y, z represent the degree to which Andy initially (before making any moves) believes that no truck, a Mexican truck, and a Lebanese truck, respectively, are located in spot B.
Example Output: [1, 4, 7]. Only output the list and do not include any explanations in the response.
Output:"""
    # print(question)
    ans, cost = gpt_request(question)
    enh_print('belief: ' + ans)
    # exit()