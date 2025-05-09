import json
import pandas as pd
import random

data_path = "../data/MMToM-QA/multimodal_representations.json"

try:
    data = []
    with open(data_path, "r") as file:
        for line in file:
            d = json.loads(line)
            data.append(d)
    print("JSON file has been successfully loaded.")
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
except FileNotFoundError:
    print(f"No file found at {data_path}")
except Exception as e:
    print(f"An error occurred: {e}")

def replace_words(s):
    replacements = [
            ('refrigerator', 'fridge'),
            ("first kitchen cabinet", "1st kitchencabinet"),
            ("second kitchen cabinet", "2nd kitchencabinet"),
            ("third kitchen cabinet", "3rd kitchencabinet"),
            ("fourth kitchen cabinet", "4th kitchencabinet"),
            ("fifth kitchen cabinet", "5th kitchencabinet"),
            ("sixth kitchen cabinet", "6th kitchencabinet"),
            ("seventh kitchen cabinet", "7th kitchencabinet"),
            ("eighth kitchen cabinet", "8th kitchencabinet"),
            ("first cabinet", "1st kitchencabinet"),
            ("second cabinet", "2nd kitchencabinet"),
            ("third cabinet", "3rd kitchencabinet"),
            ("fourth cabinet", "4th kitchencabinet"),
            ("fifth cabinet", "5th kitchencabinet"),
            ("sixth cabinet", "6th kitchencabinet"),
            ("seventh cabinet", "7th kitchencabinet"),
            ("eighth cabinet", "8th kitchencabinet"),
            ("kitchen table", "kitchentable"),
            ("bathroom cabinet", "bathroomcabinet"),
            ("condiment bottle", "condimentbottle"),
            ("remote control", "remotecontrol"),
            ("water glass", "waterglass"),
            ("wine glass", "wineglass"),
            ("walk towards", "walktowards"),
            ("coffee table", "coffeetable"),
            ("living room", "livingroom"),
            ("kitchen cabinet", "kitchencabinet"),
            ("dish bowl", "dishbowl"),
            ("third", "3rd"),
            ("second", "2nd"),
            ("first", "1st"),
            ("fourth", "4th"),
            ("fifth", "5th"),
            ("sixth", "6th")
        ]
    for old, new in replacements:
        if isinstance(s, list):
            for i, _ in enumerate(s):
                s[i] = s[i].replace(new, old)
        else:
            s = s.replace(new, old)
    return s

records = []
for d in data:
    timesteps = d["end_time"] + 1

    actions = d["actions"]
    states = []
    for i in range(timesteps + 1):
        states.append(d[f"state_{i}"])
    
    text = d["question"]
    story = text.split('\nQuestion: ')[0].strip()
    question = text.split('\nQuestion: ')[1].split(" (a) ")[0]
    choices = [
        text.split('\nQuestion: ')[1].split(" (a) ")[1].split(" (b) ")[0].strip(),
        text.split('\nQuestion: ')[1].split(" (a) ")[1].split(" (b) ")[1].split(" Please")[0].strip()
    ]
    if d["answer"] == "a":
        gt_answer = choices[0]
    else:
        gt_answer = choices[1]

    agent_name = actions[0].split(' ')[0]
    new_actions = []
    for action in actions:
        if agent_name not in action:
            action = f"{agent_name} {action}"
        new_actions.append(action)

    updated_actions = []
    diff_action_idx = []
    for j, a in enumerate(new_actions):
        if j < len(actions) - 1 and actions[j] == actions[j + 1]:
            continue
        diff_action_idx.append(j)
        a = replace_words(a)
        updated_actions.append(a)

    updated_states = []
    for j, s in enumerate(states):
        if j not in diff_action_idx:
            continue
        s = replace_words(s)
        updated_states.append(s)

    record = {
        "story": replace_words(story),
        "question": replace_words(question),
        "answer_choices": replace_words(choices),
        "gt_answer": replace_words(gt_answer),
        "states": f"{updated_states}",
        "actions": f"{updated_actions}"
    }

    assert len(actions) == len(states)
    records.append(record)

output_path_base = "../full_data_formatted/"
df = pd.DataFrame(records)
df.to_csv(f"{output_path_base}MMToM-QA.csv", index=False)

print("Data has been saved to respective files.")
