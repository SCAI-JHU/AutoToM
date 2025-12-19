from utils import llm_request
import csv
from ElementExtractor import *
from DataLoader import *
import os
from copy import deepcopy


class TimeLine:
    def __init__(
        self,
        story,
        question,
        choices,
        variable_names,
        model_name,
        episode_name,
        inf_var,
        llm,
        dataset_name,
    ):
        self.variable_names = deepcopy(variable_names)
        self.story = story
        self.agents = None
        self.llm = llm
        self.inferred_agent = None
        self.choices = choices
        self.question = question

        self.inf_var = inf_var
        self.model_name = model_name
        self.episode_name = episode_name
        self.dataset_name = dataset_name

    def extract(self, agents=None, inferred_agent=None, verbose=False):
        if agents is None:
            self.agents = find_agents(self.story, self.llm)
        else:
            self.agents = agents
        if inferred_agent is None:
            self.inferred_agent = find_inferred_agent(
                self.question, self.choices, self.llm
            )
        else:
            self.inferred_agent = inferred_agent
        # original wording
        if "BigToM" in self.dataset_name:
            with open(
                f"prompts/prompts_{self.llm}/find_inf_actions_bigToM.txt",
                "r",
                encoding="utf-8",
            ) as prompt_file:
                prompt_template = prompt_file.read().strip()
            prompt = prompt_template.replace("[Story]", f"Story: {self.story}")
            prompt = prompt.replace("[Inferred_agent]", f"{self.inferred_agent}")
            resp, cost = llm_request(prompt, temperature=0.0, model=self.llm)
            action_sequence = get_list_from_str(resp)
            if len(action_sequence) == 0:
                no_actions = True
            else:
                no_actions = False
        else:
            with open(
                f"prompts/prompts_{self.llm}/find_inf_actions.txt",
                "r",
                encoding="utf-8",
            ) as prompt_file:
                prompt_template = prompt_file.read().strip()
            prompt = prompt_template.replace("[Story]", f"Story: {self.story}")
            prompt = prompt.replace("[Inferred_agent]", f"{self.inferred_agent}")
            resp, cost = llm_request(prompt, temperature=0.0, model=self.llm)
            action_sequence = get_list_from_str(resp)
        # No actions in the story
        if action_sequence == []:
            action_sequence = [self.story]
            action_sequence_wording = action_sequence
        # At least one action in the story
        else:
            # refined wording
            with open(
                f"prompts/prompts_{self.llm}/find_inf_actions_wording.txt",
                "r",
                encoding="utf-8",
            ) as prompt_file:
                prompt_template = prompt_file.read().strip()
            prompt = prompt_template.replace("[Story]", f"Story: {self.story}")
            prompt = prompt.replace("[Inferred_agent]", f"{self.inferred_agent}")
            prompt = prompt.replace("[bare_actions]", f"{action_sequence}")
            resp, cost = llm_request(prompt, temperature=0.0, model=self.llm)
            action_sequence_wording = get_list_from_str(resp)
            # print('extracted action sequence:', action_sequence_wording)
            if "BigToM" not in self.dataset_name:
                for i, act in enumerate(action_sequence_wording):
                    with open(
                        f"prompts/prompts_{self.llm}/refine_wording_action_sequence.txt",
                        "r",
                        encoding="utf-8",
                    ) as prompt_file:
                        prompt_template = prompt_file.read().strip()
                    prompt = prompt_template.replace("[Sentence]", f"{act}")
                    prompt = prompt.replace(
                        "[Inferred_agent]", f"{self.inferred_agent}"
                    )
                    # print(prompt)
                    resp, cost = llm_request(prompt, temperature=0.0, model=self.llm)
                    # print(resp)
                    action_sequence_wording[i] = resp
            else:
                action_sequence_wording = action_sequence

        known_goal = "NONE"

        chunks = []
        chunks_wording = []
        now_story = deepcopy(self.story)
        # assert len(action_sequence) == len(
        #     action_sequence_wording
        # )  # need to add more safety checks
        if verbose:
            print("ACTION SEQUENCE")
            print(action_sequence)
            print("ACTION SEQUENCE WORDING")
            print(action_sequence_wording)

        now_story = now_story.replace("\n", " ")
        # action_sequence is the timeline
        translator = str.maketrans("", "", string.punctuation)  # get rid of punctuation
        counter = 0
        # print(len(action_sequence))
        for action in action_sequence:
            # action = action.replace(",", "").replace(".", "")
            if action[-1] in [",", "."]:
                action = action[:-1]
            try:
                # better wording
                if action_sequence_wording[counter] in now_story:
                    # enh_print(now_story.split(action_sequence_wording[counter])[0].split('. ')[-1], 'red')
                    previous_information_till_action = (
                        ". ".join(
                            now_story.split(action_sequence_wording[counter])[0].split(
                                ". "
                            )[:-1]
                        )
                        + "."
                    )
                    now_story = now_story[
                        now_story.find(action_sequence_wording[counter])
                        + len(action_sequence_wording[counter]) :
                    ]
                else:
                    previous_information_till_action = (
                        ". ".join(now_story.split(action)[0].split(". ")[:-1]) + "."
                    )
                    now_story = now_story[now_story.find(action) + len(action) :]
                if verbose:
                    enh_print(previous_information_till_action)
                chunk_wording = (
                    previous_information_till_action.strip()
                    + " "
                    + action_sequence_wording[counter]
                )
                if ".\n" in chunk_wording[:5]:
                    chunk_wording.replace(".\n", "")
                if ". " in chunk_wording[:2]:
                    chunk_wording = chunk_wording[2:]
                chunks_wording.append(chunk_wording)
                if verbose:
                    enh_print(chunk_wording, "red")
                if len(now_story) > 0:
                    if now_story[0] == "." or now_story[0] == ",":
                        now_story = now_story[1:]
                    if now_story[:2] == ". " or now_story[:2] == ", ":
                        now_story = now_story[2:]
                    now_story = now_story.strip()
                    if verbose:
                        print("now story:", now_story, "\n")
            except Exception as e:
                print(f"Error when processing action {action} in {now_story}")
            counter += 1
        now_story = now_story.strip()
        if "MuMa" in self.dataset_name:
            if len(chunks_wording) == 0:
                chunks_wording.append(now_story)
            else:
                chunks_wording[-1] += " " + now_story
        elif "ToMi" in self.dataset_name:
            chunks_wording.append(now_story)
        elif (
            ("Action" not in self.variable_names)
            or (self.inf_var == "Action")
            or (self.model_name == "automated" and now_story != "")
        ):
            # If not inferring action /model discovery, more timestep
            chunks_wording.append(now_story)
        else:
            if len(chunks_wording) > 0:
                chunks_wording[-1] += " " + now_story
            else:
                chunks_wording.append(now_story)

        if verbose:
            print(chunks)
            print()
            print(chunks_wording)
            print()
        chunks = chunks_wording
        if "BigToM" in self.dataset_name:
            chunks = [x for x in chunks_wording if x != ""]
        dicts = []
        if "Action" not in self.variable_names:
            self.variable_names.append("Action")
        for chunk in chunks:
            var_dict = {}
            var_dict["Chunk"] = chunk
            for var_name in self.variable_names:
                if verbose:
                    enh_print(chunk, color="blue")
                if var_name == self.inf_var:
                    var_dict[f"{self.inferred_agent}'s {var_name}"] = "NONE"
                elif var_name not in ["Action"]:
                    resp = extraction(
                        chunk,
                        self.inferred_agent,
                        var_name,
                        self.llm,
                        self.dataset_name,
                    )
                    if var_name == "State":
                        var_dict["State"] = parse_extraction(resp)

                    else:
                        resp = parse_extraction(resp)
                        var_dict[f"{self.inferred_agent}'s {var_name}"] = resp
                        if var_name == "Goal" and resp != "NONE":
                            known_goal = resp
                else:  # Action extraction
                    for agent in self.agents:
                        resp = extraction(
                            chunk, agent, var_name, self.llm, self.dataset_name
                        )
                        var_dict[f"{agent}'s {var_name}"] = parse_extraction(resp)
            dicts.append(var_dict)

        for d in dicts:
            all_actions = []
            for key, val in d.items():
                if "Action" in key and val != "NONE":
                    all_actions.append(val)
            d["All Actions"] = all_actions if all_actions != [] else "NONE"

        if known_goal != "NONE":
            for i, r in enumerate(dicts):
                var_name = f"{self.inferred_agent}'s Goal"
                r[var_name] = known_goal

        if verbose:
            print(dicts)
        self.save_timeline_table(dicts)
        if "BigToM" not in self.dataset_name:
            no_actions = False
        return dicts, no_actions

    def supply_extraction(self, dicts, verbose=False):
        # Supply variable extraction for automated model (the original dicts is from other models)
        self.inferred_agent = find_inferred_agent(self.question, self.choices, self.llm)
        known_goal = "NONE"
        for var_dict in dicts:
            for var_name in self.variable_names:
                chunk = var_dict["Chunk"]
                if var_name == self.inf_var:
                    var_dict[f"{self.inferred_agent}'s {var_name}"] = "NONE"
                elif var_name not in ["Action"]:
                    if var_name == "State":
                        continue

                    if f"{self.inferred_agent}'s {var_name}" in var_dict:
                        continue

                    resp = extraction(
                        chunk,
                        self.inferred_agent,
                        var_name,
                        self.llm,
                        self.dataset_name,
                    )

                    resp = parse_extraction(resp)
                    var_dict[f"{self.inferred_agent}'s {var_name}"] = resp
                    if var_name == "Goal" and resp != "NONE":
                        known_goal = resp
        if known_goal != "NONE":
            for i, r in enumerate(dicts):
                var_name = f"{self.inferred_agent}'s Goal"
                r[var_name] = known_goal
        if verbose:
            print(dicts)
        self.save_timeline_table(dicts)
        print(len(dicts))
        print(dicts[0])
        if "BigToM" in self.dataset_name:
            if (len(dicts) == 1) and f"{self.inferred_agent}'s Action" not in dicts[0]:
                new_dicts = deepcopy(dicts)
                for k in dicts[0].keys():
                    if "'s Action" in k and self.inferred_agent in k:
                        new_dicts[0][f"{self.inferred_agent}'s Action"] = dicts[0][k]
                dicts = new_dicts

            if (
                len(dicts) == 1
                and dicts[0][f"{self.inferred_agent}'s Action"] == "NONE"
            ):
                no_actions = True
            else:
                no_actions = False
        else:
            no_actions = False
        return dicts, no_actions

    def save_timeline_table(self, dicts):
        output_folder = "../results/middle"
        output_file = f"{output_folder}/{self.model_name}_{self.episode_name}.csv"

        os.makedirs(output_folder, exist_ok=True)
        with open(output_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=dicts[0].keys())
            writer.writeheader()
            for dict in dicts:
                writer.writerow(dict)


def load_timeline_table(model_name, episode_name, reuse=False):
    file_name = f"../results/middle/{model_name}_{episode_name}.csv"

    if not os.path.isfile(file_name):
        if reuse is True:
            file_name = get_filename_with_episode_name(
                episode_name=episode_name, base_path="../results/middle/"
            )
            if file_name is None:
                return None
        else:
            return None
    dicts = []
    with open(file_name, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dicts.append(dict(row))
    return dicts
