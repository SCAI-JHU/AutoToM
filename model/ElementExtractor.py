import numpy as np
from utils import *
from probs import *
from copy import deepcopy
import json


class Variable:
    def __init__(
        self,
        name,
        in_model=False,
        is_observed=False,
        possible_values=None,
        prior_probs=None,
    ):
        self.name = name
        self.in_model = in_model
        self.is_observed = is_observed
        self.possible_values = possible_values
        self.prior_probs = prior_probs

    def __repr__(self):
        return f'Variable(name="{self.name}", in_model={self.in_model}, is_observed={self.is_observed}, possible_values={self.possible_values}, prior_probs={self.prior_probs})'


def get_base64_images(video_id, frames):
    extract_frames(video_id, frames)
    images_base64 = []
    for i in range(frames):
        image_path = f"../benchmarks/data/MuMa/frames/{video_id}_frame_{i}.png"
        img = encode_image(image_path)
        images_base64.append(img)
    return images_base64


def get_story_from_video(images_base64, llm):
    prompt = """Extract the actions of the agents in the video. Only contain actions that you're sure about and do not imagine any actions.
Agents should have name. Formulate your response in a single line. Actions contain walking, speaking and moving objects. """
    story, cost = gpt_request_multimodal(prompt, images_base64, model=llm)
    # print(story)
    quit()


def guided_state_filter(state, information, llm, relevant_entities=None):
    if relevant_entities is not None:
        flag = False
        for ent in relevant_entities:
            if ent in state:
                flag = True
        return flag
    with open(
        f"prompts/prompts_{llm}/guided_belief_of_state.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Information]", f"{information}")
    prompt = prompt.replace("[Sentence]", state)

    resp, cost = llm_request(prompt, temperature=0.0, hypo=True, model=llm)

    if resp[0] == "B":
        return False
    else:
        return True


def generate_hypo_belief_of_state(s, c, llm):
    with open(
        f"prompts/prompts_{llm}/hypo_Belief_of_State.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    c_new = deepcopy(c)
    # print(c_new)
    for i in range(len(c)):
        c_new[i] = c[i].split("the ")[-1].replace(".", "")
    prompt = prompt_template.replace("[Information]", f"Information: {c_new}")
    prompt = prompt.replace("[Given Sentence]", f"{s}")
    resp, cost = llm_request(prompt, temperature=0.0, hypo=True, model=llm)
    print(prompt)
    # print('before handling', resp)
    resp_list = get_list_from_str(resp)
    print("after converting", resp_list)
    return resp_list


def get_belief_of_states(states, agent, choices, llm, relevant_entities=None):
    single_states = states.split(".")
    beliefs = []
    for s in single_states:
        s = s.strip()
        if s != "" and s != "NONE":
            flag = False
            for c in choices:
                if guided_state_filter(s, c, llm, relevant_entities):
                    flag = True

            if flag:
                opposite_sentences = generate_hypo_belief_of_state(s, choices, llm)
                beliefs.append(opposite_sentences)
            else:
                print("Discarded state:", s)
    return beliefs


def video_extracted_actions(story_id):
    with open("../benchmarks/data/MuMa/gemini_outputs.json") as file:
        action = json.load(file)[str(story_id)]["action"]

    return action


def hypothesis_generation_no_observation(info, character, llm, verbose=False):
    with open(
        f"prompts/prompts_{llm}/hypo_Observation_no_observation.txt",
        "r",
        encoding="utf-8",
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Character]", f"{character}")
    prompt = prompt.replace("[Information]", f"{info}")

    resp, cost = llm_request(prompt, temperature=0.0, hypo=True, model=llm)
    # enh_print(prompt, resp)

    if "\n" in resp:
        resp = resp.split("\n")[-1]
    if resp[0] != "[" and "[" in resp:
        resp = "[" + resp.split("[")[1]
    elif resp[-1] != "]" and "]" in resp:
        resp = resp.split("]")[0] + "]"
    if '"' not in resp:
        resp = resp.split("[")[1]
        resp = resp.split("]")[0]
        resp = f'["{resp}"]'

    resp_list = eval(resp)
    # print(resp_list)
    # print(res)
    # if verbose:
    # enh_print(f"Hypotheses proposed for no observation: {resp_list}")
    return resp_list


def repetitive_hypothesis_reduction(hypo_c, llm):
    with open(
        f"prompts/prompts_{llm}/repetitive_hypo_reduction.txt",
        "r",
        encoding="utf-8",
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
        prompt = prompt_template.replace("[Hypotheses]", f"{hypo_c}")
        resp, cost = llm_request(prompt, temperature=0.0, hypo=True, model=llm)
    # enh_print(prompt, resp)
    # print(prompt, resp)
    try:
        return eval(resp)
    except Exception as e:
        return hypo_c


def hypothesis_generation(
    wrong_hyp,
    info,
    story,
    character,
    element_name,
    K,
    llm,
    verbose=False,
    prev_hyp="",
    dataset_name="",
):
    if element_name == "Belief" and "BigToM" in dataset_name:
        with open(
            f"prompts/prompts_{llm}/hypo_{element_name}_BigToM.txt",
            "r",
            encoding="utf-8",
        ) as prompt_file:
            prompt_template = prompt_file.read().strip()

    else:
        with open(
            f"prompts/prompts_{llm}/hypo_{element_name}.txt", "r", encoding="utf-8"
        ) as prompt_file:
            prompt_template = prompt_file.read().strip()

    prompt = prompt_template.replace("[Context]", f"Story: {story}")
    prompt = prompt.replace("[Character]", f"{character}")
    prompt = prompt.replace("[Wrong Hypotheses]", f"{wrong_hyp}")
    prompt = prompt.replace("[Information]", f"{info}")
    if K != 1:
        prompt = prompt.replace("[num]", f"{K}")
    else:
        prompt = prompt.replace("[num]", f"one")
        prompt = prompt.replace("hypotheses", "hypothesis")
        prompt = prompt.replace("Hypotheses", "Hypothesis")
        prompt = prompt.replace("align", "aligns")
        prompt = prompt.replace('["aaa.", "bbb.", ...]', '["aaa."]')

    resp, cost = llm_request(prompt, temperature=0.0, hypo=True, model=llm)

    if "\n" in resp:
        resp = resp.split("\n")[-1]
    if resp[0] != "[" and "[" in resp:
        resp = "[" + resp.split("[")[1]
    elif resp[-1] != "]" and "]" in resp:
        resp = resp.split("]")[0] + "]"
    if '"' not in resp:
        resp = resp.split("[")[1]
        resp = resp.split("]")[0]
        resp = f'["{resp}"]'

    resp_list = eval(resp)
    for j, resp in enumerate(resp_list):
        for i in range(K):
            resp = resp.replace(f"Hypothesis_{i+1}: ", "")
        resp_list[j] = resp
    # print(resp_list)
    # print(res)
    print(prompt)
    enh_print(f"Hypotheses proposed for {element_name}\n{resp_list}")
    return resp_list


def extraction(story, character, element_name, llm, dataset_name, choices=None):
    if "BigToM" in dataset_name and element_name in [
        "Observation",
        "State",
        "Action",
        "Goal",
    ]:

        with open(
            f"prompts/prompts_{llm}/find_{element_name}_BigToM.txt",
            "r",
            encoding="utf-8",
        ) as prompt_file:
            prompt_template = prompt_file.read().strip()

        if element_name == "Observation":
            prompt = prompt_template.replace("[Story]", f"Story: {story}")
            prompt = prompt.replace("[Character]", f"{character}")
            resp, cost = llm_request(prompt, temperature=0.0, model=llm)
            # No Observation Present

            # Observation Present: See if there are more details needed for the observation
            if (
                "BigToM" in dataset_name
                and element_name in ["Observation"]
                and resp not in ['["A", ""]', '["B", ""]']
            ):
                observation = eval(resp)[1]
                with open(
                    f"prompts/prompts_{llm}/find_Observation_BigToM_extra_info.txt",
                    "r",
                    encoding="utf-8",
                ) as prompt_file:
                    prompt_template = prompt_file.read().strip()
                    prompt = prompt_template.replace("[Story]", f"Story: {story}")
                    prompt = prompt.replace("[Character]", f"{character}")
                    prompt = prompt.replace("[Observation]", f"{observation}")
                additional_resp, cost = llm_request(prompt, temperature=0.0, model=llm)
                additional_resp = eval(additional_resp)
                resp = eval(resp)

                if resp[1] not in additional_resp[1]:
                    additional_resp[1] = additional_resp[1] + " " + resp[1]
                resp[1] = additional_resp[1]
                return resp

    else:
        try:
            with open(
                f"prompts/prompts_{llm}/find_{element_name}.txt", "r", encoding="utf-8"
            ) as prompt_file:
                prompt_template = prompt_file.read().strip()
        except FileNotFoundError:
            return ["B", ""]
    prompt = prompt_template.replace("[Story]", f"Story: {story}")
    if "BigToM" in dataset_name and element_name in ["Goal"]:
        prompt = prompt.replace("[Information]", f"{choices}")
    prompt = prompt.replace("[Character]", f"{character}")
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    # check variable
    try:
        resp = eval(resp)

        if (
            "BigToM" in dataset_name and element_name in ["Observation", "State"]
        ) and resp == ["A", ""]:
            resp = ["B", ""]
            return resp
        if (
            "BigToM" in dataset_name
            and element_name not in ["State"]
            and resp != ["B", ""]
        ):
            check_var = verify_variable(element_name, resp)
        else:
            check_var = "A"
        if check_var == "A":
            return resp
        else:
            return ["B", ""]
    except Exception:
        return ["B", ""]


def get_context(story, llm):
    with open(
        f"prompts/prompts_{llm}/find_context.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Story]", f"Story: {story}")
    # print(prompt)
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    # print(resp)
    return resp


def get_initial_state(
    story, llm
):  # Note: This is for MuMAToM modality fusion, using the original LIMP prompt
    with open(
        f"prompts/prompts_{llm}/find_initial_states.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Story]", f"Story: {story}")
    # print(prompt)
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    # print(resp)
    # quit()
    return resp


def get_initial_state_tomi(story, llm):
    with open(
        f"prompts/prompts_{llm}/find_initial_state_ToMi.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Story]", f"Story: {story}")
    # print(prompt)
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    # print(resp)
    # quit()
    return resp


def verify_variable(infer_variable, sentence):
    with open(
        f"prompts/prompts_gpt-4o/determine_variable.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Sentence]", f"{sentence}")
    prompt = prompt.replace("[Variable]", infer_variable)
    resp, cost = llm_request(prompt, temperature=0.0, model="gpt-4o")
    return resp

def verify_hypo(gt_hypo, gen_hypo):
    with open(
        f"prompts/prompts_gpt-4o/verify_hypo.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Sentence1]", f"{gt_hypo}")
    prompt = prompt.replace("[Sentence2]", f"{gen_hypo}")
    resp, cost = llm_request(prompt, temperature=0.0, model="gpt-4o")
    # enh_print(prompt + resp)
    if "A" in resp[:3]:
        return gt_hypo
    else:
        return gt_hypo + ' ' + gen_hypo

def get_inf_var(question, choices, model, llm, dataset_name):
    if "belief_of_goal" in dataset_name:
        return "Belief of Goal"
    if "_belief" in dataset_name[-6:]:
        return "Belief"
    if "_goal" in dataset_name[-5:]:
        return "Goal"
    with open(
        f"prompts/prompts_{llm}/find_Inferred_Var.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Choices]", f"Choices: {choices}")

    variables = []
    for var in ["Belief", "Goal", "Action"]:
        if var in model:
            variables.append(var)
    variables = ", ".join(variables)
    variables += "."
    prompt = prompt.replace("[Variables]", variables)
    # print(prompt)
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    # print(resp)
    resp = resp.replace(".", "")
    return resp


def get_info_from_question(question, llm, dataset_name):

    with open(
        f"prompts/prompts_{llm}/get_info_from_question.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Question]", f"Question: {question}")
    # print(prompt)
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    resp = resp.replace("NONE", "")
    # print(resp)
    return resp


def get_actions_without_inf_agent(vals, inf_agent_name, full):
    if isinstance(vals["All Actions"], str):
        if vals["All Actions"] == "NONE":
            all_actions = []
        else:
            all_actions = eval(vals["All Actions"])
    else:
        all_actions = vals["All Actions"]

    inf_agent_action = "NONE"
    if f"{inf_agent_name}'s Action" in vals:
        inf_agent_action = vals[f"{inf_agent_name}'s Action"]

    if inf_agent_action in all_actions:
        all_actions.remove(inf_agent_action)

    elif (
        not full
    ):  # We don't want the last action to weigh in here -- used in solving nested belief of states
        now_chunk = vals["Chunk"]
        last_action_position = -1
        last_action = "NONE"
        for act in all_actions:
            if now_chunk.find(act) > last_action_position:
                last_action_position = now_chunk.find(act)
                last_action = act
        if last_action in all_actions:
            all_actions.remove(last_action)

    actions_without_inf_agent = " ".join(all_actions)
    return actions_without_inf_agent, inf_agent_action


def mmtom_get_variables(
    val_with_time,
    variable_types,
    inf_agent_name,
    inf_var_name,
    context,
    choices,
    K,
    llm,
    hypo_llm,
    world_rules,
    verbose,
    hypo_method,
    dataset_name,
    full,
    init_state="NONE",
    states=None,
    actions=None,
    question=None,
):
    if verbose:
        print(choices)
    res = []
    now_story = ""
    last_state = init_state
    known_Goal = "NONE"
    known_Belief = "NONE"
    inf_agent_action = "NONE"

    if "has been trying" in question:
        known_Goal = question.split("If ")[1].split(",")[0] + "."
    elif "doesn't think" in question:
        known_Belief = question.split("If ")[1].split(",")[0] + "."

    val_with_time = []
    for i in range(len(actions)):
        if inf_agent_name not in actions[i]:
            actions[i] = f"{inf_agent_name} {actions[i]}."
        vals = {
            f"{inf_agent_name}'s Action": actions[i],
            "State": states[i],
            f"{inf_agent_name}'s Belief": known_Belief,
            f"{inf_agent_name}'s Goal": known_Goal,
            f"{inf_agent_name}'s Observation": "NONE",
            "All Actions": [actions[i]],
        }
        val_with_time.append(vals)

    for i, vals in enumerate(val_with_time):
        var_dict = {}
        for var_type in variable_types:

            if var_type[1] == "State":
                var_dict["State"] = Variable(
                    name="State",
                    in_model=True,
                    is_observed=True,
                    possible_values=[vals["State"]],
                )
                continue
            if var_type[0] != "":
                var_name = f"{var_type[0]}'s {var_type[1]}"
                character = var_type[0]
            else:
                var_name = var_type[1]
                character = inf_agent_name

            if var_type[0] == inf_agent_name and var_type[1] == inf_var_name:
                var_name = f"{var_type[0]}'s {var_type[1]}"
                var_dict[var_name] = Variable(
                    name=var_name,
                    in_model=True,
                    is_observed=False,
                    possible_values=choices,
                )
                continue

            if vals[var_name] != "NONE":
                if "Belief" not in var_name:
                    hypos = [vals[var_name]]
                    var_dict[var_name] = Variable(
                        name=var_name,
                        in_model=True,
                        is_observed=True,
                        possible_values=hypos,
                    )
                else:
                    # Because a given belief hypotheses is definitely not covering the whole belief set of the agent
                    hypos = []
                    for c in choices:
                        gen_hypo = hypothesis_generation(
                            [], c + '; ' + vals[f"{inf_agent_name}'s Action"], now_story, character, var_type[1], K, hypo_llm
                        )[0]
                        hypos.append(verify_hypo(vals[var_name], gen_hypo))
                            
                    var_dict[var_name] = Variable(
                        name=var_name,
                        in_model=True,
                        is_observed=False,
                        possible_values=hypos,
                    )
            else:

                if var_type[1] == "Observation":
                    hypos = []
                    # Propose quality hypotheses for observation and belief, conditioned on observable variables
                    for c in choices:
                        hypos += hypothesis_generation([], c + '; ' + vals[f"{inf_agent_name}'s Action"], 
                            now_story, inf_agent_name,"Observation",1,hypo_llm,
                        )
                    hypos += hypothesis_generation_no_observation(
                        choices, inf_agent_name, hypo_llm, True
                    )

                elif var_type[1] == "Goal" and known_Goal != "NONE":
                    var_dict[var_name] = Variable(
                        name=var_name,
                        in_model=True,
                        is_observed=True,
                        possible_values=[known_Goal],
                    )
                    continue
                else:
                    hypos = []
                    for c in choices:
                        hypos += hypothesis_generation(
                            [], c + '; ' + vals[f"{inf_agent_name}'s Action"], now_story, character, var_type[1], K, hypo_llm
                        )

                var_dict[var_name] = Variable(
                    name=var_name,
                    in_model=True,
                    is_observed=False,
                    possible_values=hypos,
                )
        res.append(var_dict)
        # print(res)
    return res


def mmtom_get_variables_at_time(
    time,
    variable_types,
    inf_agent_name,
    inf_var_name,
    choices,
    K,
    llm,
    hypo_llm,
    verbose,
    hypo_method,
    full,
    preproposed_ob_hypos,
    states=None,
    actions=None,
    question=None,
):
    if verbose:
        print(choices)
    res = []
    now_story = ""
    known_Goal = "NONE"
    known_Belief = "NONE"

    if "has been trying" in question:
        known_Goal = question.split("If ")[1].split(",")[0] + "."
    elif "doesn't think" in question:
        known_Belief = question.split("If ")[1].split(",")[0] + "."

    vals = {
        f"{inf_agent_name}'s Action": actions[time],
        "State": states[time],
        f"{inf_agent_name}'s Belief": known_Belief,
        f"{inf_agent_name}'s Goal": known_Goal,
        f"{inf_agent_name}'s Observation": "NONE",
        "All Actions": [actions[time]],
    }

    var_dict = {}
    for var_type in variable_types:

        if var_type[1] == "State":
            var_dict["State"] = Variable(
                name="State",
                in_model=True,
                is_observed=True,
                possible_values=[vals["State"]],
            )
            continue
        if var_type[0] != "":
            var_name = f"{var_type[0]}'s {var_type[1]}"
            character = var_type[0]
        else:
            var_name = var_type[1]
            character = inf_agent_name

        if var_type[0] == inf_agent_name and var_type[1] == inf_var_name:
            var_name = f"{var_type[0]}'s {var_type[1]}"
            var_dict[var_name] = Variable(
                name=var_name,
                in_model=True,
                is_observed=False,
                possible_values=choices,
            )
            continue

        if vals[var_name] != "NONE":
            if "Belief" not in var_name:
                hypos = [vals[var_name]]
                var_dict[var_name] = Variable(
                    name=var_name,
                    in_model=True,
                    is_observed=True,
                    possible_values=hypos,
                )
            else:
                # Because a given belief hypotheses is definitely not covering the whole belief set of the agent
                hypos = []
                for c in choices:
                    gen_hypo = hypothesis_generation(
                        [], c + '; ' + vals[f"{inf_agent_name}'s Action"], now_story, character, var_type[1], K, hypo_llm
                    )[0]
                    hypos.append(verify_hypo(vals[var_name], gen_hypo))
                        
                var_dict[var_name] = Variable(
                    name=var_name,
                    in_model=True,
                    is_observed=False,
                    possible_values=hypos,
                )
        else:
            # We don't propose hypotheses for actions
            # We don't propose hypotheses for utterances
            if var_type[1] == "Action" or var_type[1] == "Utterance":
                var_dict[var_name] = Variable(
                    name=var_name,
                    in_model=True,
                    is_observed=True,
                    possible_values=["NONE"],
                )
                continue
            if var_type[1] == "Observation":
                hypos = []
                # Propose quality hypotheses for observation and belief, conditioned on observable variables
                for c in choices:
                    hypos += hypothesis_generation([], c + '; ' + vals[f"{inf_agent_name}'s Action"], 
                        now_story, inf_agent_name,"Observation",1,hypo_llm,
                    )
                hypos += hypothesis_generation_no_observation(
                    choices, inf_agent_name, hypo_llm, True
                )

            elif var_type[1] == "Goal" and known_Goal != "NONE":
                var_dict[var_name] = Variable(
                    name=var_name,
                    in_model=True,
                    is_observed=True,
                    possible_values=[known_Goal],
                )
                continue
            else:
                hypos = []
                for c in choices:
                    hypos += hypothesis_generation(
                        [], c + '; ' + vals[f"{inf_agent_name}'s Action"], now_story, character, var_type[1], K, hypo_llm
                    )
            var_dict[var_name] = Variable(
                name=var_name,
                in_model=True,
                is_observed=False,
                possible_values=hypos,
            )

    return var_dict, preproposed_ob_hypos


def get_variables_at_time(
    time,
    vals,
    variable_types,
    inf_agent_name,
    inf_var_name,
    choices,
    K,
    llm,
    hypo_llm,
    verbose,
    hypo_method,
    full,
    preproposed_ob_hypos,
    last_state,
    inf_agent_action,
    dataset_name,
    precomputed_states,
    predefined_bel_hypos,
):
    now_story = vals["Chunk"]
    inf_agent_action = inf_agent_action

    var_dict = {}
    for var_type in variable_types:
        if var_type[1] == "State":
            now_state = precomputed_states[time]
            var_dict["State"] = Variable(
                name="State",
                in_model=True,
                is_observed=True,
                possible_values=[now_state],
            )
            continue

        if var_type[1] == "All Actions":
            var_dict["All Actions"] = Variable(
                name="All Actions",
                in_model=True,
                is_observed=True,
                possible_values=[vals["All Actions"]],
            )
            continue

        if var_type[0] != "":
            var_name = f"{var_type[0]}'s {var_type[1]}"
            character = var_type[0]
        else:
            var_name = var_type[1]
            character = inf_agent_name

        if var_type[0] == inf_agent_name and var_type[1] == inf_var_name:
            var_name = f"{var_type[0]}'s {var_type[1]}"
            var_dict[var_name] = Variable(
                name=var_name,
                in_model=True,
                is_observed=False,
                possible_values=choices,
            )
            continue

        if var_name in vals and vals[var_name] != "NONE":
            var_dict[var_name] = Variable(
                name=var_name,
                in_model=True,
                is_observed=True,
                possible_values=[vals[var_name]],
            )
        else:
            # We don't propose hypotheses for actions
            # We don't propose hypotheses for utterances
            if var_type[1] == "Action" or var_type[1] == "Utterance":
                var_dict[var_name] = Variable(
                    name=var_name,
                    in_model=True,
                    is_observed=True,
                    possible_values=["NONE"],
                )
                continue

            if (
                var_type[1] == "Observation"
            ):  # also change the way we propose hyp for belief
                if hypo_method == "guided":
                    if len(preproposed_ob_hypos) == 0:
                        for c in choices:
                            preproposed_ob_hypos += hypothesis_generation(
                                [],
                                c,
                                now_story,
                                character,
                                var_type[1],
                                1,
                                hypo_llm,
                            )
                        preproposed_ob_hypos += hypothesis_generation_no_observation(
                            choices, character, hypo_llm, True
                        )
                    hypos = preproposed_ob_hypos
            elif var_type[1] == "Belief" and predefined_bel_hypos != None:
                hypos = predefined_bel_hypos
            else:
                hypo_c = []
                for c in choices:

                    hypo_c += hypothesis_generation(
                        [], c, now_story, character, var_type[1], K, hypo_llm
                    )

                hypos = hypo_c
            var_dict[var_name] = Variable(
                name=var_name,
                in_model=True,
                is_observed=False,
                possible_values=hypos,
            )

    return var_dict, preproposed_ob_hypos, now_state, inf_agent_action


def get_variables_with_time(
    val_with_time,
    variable_types,
    inf_agent_name,
    inf_var_name,
    context,
    choices,
    K,
    llm,
    hypo_llm,
    world_rules,
    verbose,
    hypo_method,
    dataset_name,
    full,
    init_state="NONE",
    prev_hyp="",
    states=None,
    actions=None,
    question=None,
):
    if "MMToM" in dataset_name:
        time_variables = mmtom_get_variables(
            val_with_time,
            variable_types,
            inf_agent_name,
            inf_var_name,
            context,
            choices,
            K,
            llm,
            hypo_llm,
            world_rules,
            verbose,
            hypo_method,
            dataset_name,
            full,
            init_state,
            states,
            actions,
            question,
        )
        return time_variables

    res = []
    now_story = ""
    last_state = init_state
    known_Goal = "NONE"
    inf_agent_action = "NONE"

    entire_story = ""
    if isinstance(val_with_time, tuple):
        val_with_time, _ = val_with_time
    for vals in val_with_time:
        entire_story += vals["Chunk"]

    preproposed_ob_hypos = []
    for i, vals in enumerate(val_with_time):
        now_story += vals["Chunk"]
        var_dict = {}

        for var_type in variable_types:
            if var_type[1] == "State":
                now_state = deepcopy(last_state)
                now_state = update_state(
                    now_state, inf_agent_action, llm, verbose, dataset_name
                )
                now_state = update_state(
                    now_state, vals["State"], llm, verbose, dataset_name
                )
                # The effect of inf_agent_action is delayed to next timestep (A_t takes place after S_t)
                actions_without_inf_agent, inf_agent_action = (
                    get_actions_without_inf_agent(vals, inf_agent_name, full)
                )

                now_state = update_state(
                    now_state, actions_without_inf_agent, llm, verbose, dataset_name
                )
                last_state = deepcopy(now_state)
                var_dict["State"] = Variable(
                    name="State",
                    in_model=True,
                    is_observed=True,
                    possible_values=[now_state],
                )
                continue

            if var_type[1] == "All Actions":
                var_dict["All Actions"] = Variable(
                    name="All Actions",
                    in_model=True,
                    is_observed=True,
                    possible_values=[vals["All Actions"]],
                )
                continue

            if var_type[0] != "":
                var_name = f"{var_type[0]}'s {var_type[1]}"
                character = var_type[0]
            else:
                var_name = var_type[1]
                character = inf_agent_name

            if var_type[0] == inf_agent_name and var_type[1] == inf_var_name:
                var_name = f"{var_type[0]}'s {var_type[1]}"
                var_dict[var_name] = Variable(
                    name=var_name,
                    in_model=True,
                    is_observed=False,
                    possible_values=choices,
                )
                continue

            if prev_hyp not in ["", None]:
                if "Observation" in var_name:
                    vals[var_name] = "NONE"

            if var_name in vals and vals[var_name] != "NONE":
                if var_type[1] == "Goal":
                    known_Goal = deepcopy(vals[var_name])
                var_dict[var_name] = Variable(
                    name=var_name,
                    in_model=True,
                    is_observed=True,
                    possible_values=[vals[var_name]],
                )
            else:
                # We don't propose hypotheses for actions
                # We don't propose hypotheses for utterances
                if var_type[1] == "Action" or var_type[1] == "Utterance":
                    if len(res) > 0:
                        var_dict[var_name] = res[-1][var_name]
                    else:
                        var_dict[var_name] = Variable(
                            name=var_name,
                            in_model=True,
                            is_observed=True,
                            possible_values=["NONE"],
                        )
                    continue

                if (
                    var_type[1] == "Observation"
                ):  # also change the way we propose hyp for belief
                    if len(preproposed_ob_hypos) == 0:
                        if var_type[1] == "Belief" and "BigToM" in dataset_name:
                            preproposed_ob_hypos += hypothesis_generation(
                                [],
                                choices,
                                now_story,
                                character,
                                var_type[1],
                                K,
                                hypo_llm,
                                prev_hyp,
                                dataset_name,
                            )
                        else:
                            for c in choices:
                                preproposed_ob_hypos += hypothesis_generation(
                                    [],
                                    c,
                                    now_story,
                                    character,
                                    var_type[1],
                                    1,
                                    hypo_llm,
                                    prev_hyp,
                                    dataset_name,
                                )
                            if prev_hyp in preproposed_ob_hypos:
                                preproposed_ob_hypos.remove(prev_hyp)

                        preproposed_ob_hypos += hypothesis_generation_no_observation(
                            choices, character, hypo_llm, True
                        )
                        enh_print(
                            f"New observation hypotheses: {preproposed_ob_hypos}",
                            "red",
                        )
                    hypos = preproposed_ob_hypos

                elif var_type[1] == "Goal" and known_Goal != "NONE":
                    var_dict[var_name] = Variable(
                        name=var_name,
                        in_model=True,
                        is_observed=True,
                        possible_values=[known_Goal],
                    )
                    continue
                else:
                    if var_type[1] == "Belief" and "BigToM" in dataset_name:
                        hypo_c = []
                        hypo_c += hypothesis_generation(
                            [],
                            choices,
                            now_story,
                            character,
                            var_type[1],
                            K,
                            hypo_llm,
                            dataset_name,
                        )
                    else:
                        hypo_c = []
                        for c in choices:

                            hypo_c += hypothesis_generation(
                                [],
                                c,
                                now_story,
                                character,
                                var_type[1],
                                1,
                                hypo_llm,
                                dataset_name,
                            )
                    hypos = hypo_c
                var_dict[var_name] = Variable(
                    name=var_name,
                    in_model=True,
                    is_observed=False,
                    possible_values=hypos,
                )
        res.append(var_dict)
    # We assume that one has a consistent goal
    if known_Goal != "NONE":
        for i, r in enumerate(res):
            var_name = f"{inf_agent_name}'s Goal"
            if r[var_name].is_observed == False:
                r[var_name] = Variable(
                    name=var_name,
                    in_model=True,
                    is_observed=True,
                    possible_values=[known_Goal],
                )
    return res


def save_time_variables(dicts, model_name, episode_name):
    output_folder = "../results/var"
    output_file = f"{output_folder}/{model_name}_{episode_name}.csv"

    os.makedirs(output_folder, exist_ok=True)
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=dicts[0].keys())
        writer.writeheader()
        for dict in dicts:
            writer.writerow(dict)
    print(f"Var results saved to {output_file}")


def save_reconstructed_story(vis, model_name, episode_name, first_agent_name):
    output_folder = "../results/nested"
    output_file = f"{output_folder}/{model_name}_{episode_name}_{first_agent_name}_reconstructed_story.csv"

    os.makedirs(output_folder, exist_ok=True)
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["Original story", "Reconstructed story"]
        )
        writer.writeheader()
        for v in vis:
            writer.writerow(v)


def save_belief_probs(probs, model_name, episode_name):
    output_folder = "../results/probs"
    output_file = f"{output_folder}/{model_name}_{episode_name}.csv"

    os.makedirs(output_folder, exist_ok=True)
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=probs[0].keys())
        writer.writeheader()
        for prob in probs:
            writer.writerow(prob)
    print(f"Probs results saved to {output_file}")


def save_metrics(metrics, model_name, episode_name, back_inference, reduce_hypos):

    output_folder = "../results/metrics"
    base_file_name = f"{model_name}_{episode_name}_back{int(back_inference)}_reduce{int(reduce_hypos)}_metrics.json"
    output_file = os.path.join(output_folder, base_file_name)

    os.makedirs(output_folder, exist_ok=True)

    count = 1
    while os.path.exists(output_file):
        new_file_name = f"{base_file_name}_{count}.json"
        output_file = os.path.join(output_folder, new_file_name)
        count += 1

    with open(output_file, mode="w") as file:
        json.dump(metrics, file, indent=2)


def save_node_results(
    node_results,
    model_name,
    episode_name,
    back_inference,
    red_obs_hypo,
):
    output_folder = "../results/node_results"
    output_file = f"{output_folder}/{model_name}_{episode_name}_back{int(back_inference)}_reduce{int(red_obs_hypo)}.csv"

    os.makedirs(output_folder, exist_ok=True)

    node_results = sorted(node_results, key=lambda x: (x["Time"], x["Node"]))

    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=node_results[0].keys())
        writer.writeheader()
        for node in node_results:
            writer.writerow(node)
    print(f"Node results saved to {output_file}")


def save_intermediate_probs(individual_likelihoods, model_name, episode_name):
    output_folder = "../results/intermediate_probs"
    output_file = f"{output_folder}/{model_name}_{episode_name}.csv"
    os.makedirs(output_folder, exist_ok=True)
    print(individual_likelihoods)

    # Prepare data for CSV
    rows = [[key[0], key[1], value] for key, value in individual_likelihoods.items()]

    # Write to CSV
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Variable", "Dependencies", "Likelihood"])
        # Write rows
        writer.writerows(rows)

    print(f"Saved intermediate probabilities to {output_file}")


def save_ipomdp_intermediate_story(story, question, choice, model_name, episode_name):
    output_folder = "../results/nested"
    output_file = (
        f"{output_folder}/{model_name}_{episode_name}_story_question_choices.csv"
    )

    res_dict = {"Story": story, "Question": question, "Choices": choice}

    os.makedirs(output_folder, exist_ok=True)
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Story", "Question", "Choices"])
        writer.writeheader()
        writer.writerow(res_dict)


def load_estimation_dict(dataset_name):
    file_name = f"../results/estimation_dicts/{dataset_name}.json"
    if not os.path.isfile(file_name):
        return {}
    with open(file_name, mode="r") as file:
        res = json.load(file)
    return res


def save_estimation_dict(dataset_name, dict):
    folder = "../results/estimation_dicts/"
    os.makedirs(folder, exist_ok=True)
    file_name = f"../results/estimation_dicts/{dataset_name}.json"
    with open(file_name, mode="w") as file:
        json.dump(dict, file, indent=2)


def save_parsed_result(info, model_name, episode_name):
    os.makedirs("../results/parsed_result", exist_ok=True)
    file_name = f"../results/parsed_result/{model_name}_{episode_name}.json"
    with open(file_name, mode="w") as file:
        json.dump(info, file, indent=2)


def load_parsed_result(model_name, episode_name, reuse=False):
    file_name = f"../results/parsed_result/{model_name}_{episode_name}.json"
    if not os.path.isfile(file_name):
        if reuse is True:
            file_name = get_filename_with_episode_name(
                episode_name=episode_name,
                base_path="../results/parsed_result/",
                suffix="json",
            )
            if file_name is None:
                return None
        else:
            return None

    with open(file_name, mode="r") as file:
        info = json.load(file)
    return info


def load_time_variables(model_name, episode_name, reuse=False):
    file_name = f"../results/var/{model_name}_{episode_name}.csv"
    if not os.path.isfile(file_name):
        if reuse is True:
            file_name = get_filename_with_episode_name(
                episode_name=episode_name, base_path="../results/var/"
            )
            if file_name is None:
                return None
        else:
            return None
    dicts = []
    with open(file_name, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            r = dict(row)
            for key, val in r.items():
                r[key] = eval(val)
            dicts.append(r)
    return dicts


def load_node_results(episode_name, back_inference, red_obs_hypo):
    output_folder = "../results/node_results"
    file_name = f"{output_folder}/{episode_name}_back{int(back_inference)}_reduce{int(red_obs_hypo)}.csv"

    if not os.path.isfile(file_name):
        return None

    dicts = []
    with open(file_name, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            r = dict(row)
            for key, val in r.items():
                if "[" in val:
                    val = eval(val)
                r[key] = val
            dicts.append(r)
    return dicts


def get_variables(story, character, var_name_list, K, inf_var_name, llm, dataset_name):
    var_list = []
    for var_name in var_name_list:
        if var_name == inf_var_name:
            continue
        resp = extraction(story, character, var_name, llm, dataset_name)
        hypos, val = None, None
        if "B) Not clearly stated" in resp:
            hypos = hypothesis_generation(story, var_name, K=K, llm=llm)
        elif "A) Clearly stated" in resp:
            val = resp.split("A) Clearly stated")[1].strip()
        else:
            if "B" in resp:
                hypos = hypothesis_generation(story, var_name, K=K, llm=llm)
            else:
                val = resp.split("A")[1].strip()
        if hypos:
            for it in hypos:
                it = it.strip().replace('"', "")
            var_list.append(
                Variable(
                    var_name, in_model=True, is_observed=False, possible_values=hypos
                )
            )
        else:
            val = val.strip().replace('"', "").replace("Evidence: ", "")
            var_list.append(
                Variable(
                    var_name, in_model=True, is_observed=True, possible_values=[val]
                )
            )
    # print(var_list)
    return var_list


def update_state(old_state, change, llm, verbose, dataset_name):
    change = change.strip()
    if change == "NONE" or change == "":
        return old_state
    with open(
        f"prompts/prompts_{llm}/update_state.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Old_State]", f"Old State: {old_state}")
    prompt = prompt.replace("[Change]", f"Change: {change}")

    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    resp = resp.replace("\n", " ")

    return resp


def load_hypos(episode_name):
    file_name = f"../results/hypos/{episode_name}.json"
    if not os.path.isfile(file_name):
        return None
    with open(file_name, mode="r") as file:
        hypos = json.load(file)
    return hypos


def save_hypos(hypos, model_name, episode_name):
    output_folder = "../results/hypos"
    output_file = f"{output_folder}/{model_name}_{episode_name}.csv"

    os.makedirs(output_folder, exist_ok=True)
    with open(output_file, mode="w", newline="") as file:
        json.dump(hypos, file)
    print(f"Hypos results saved to {output_file}")


def get_answer_from_state(state, choices, llm):
    probs = []
    state = "Ground truth state: " + state
    for c in choices:
        probs.append(
            get_likelihood_general(state, c, llm, variable="Real", verbose=True)
        )
    probs = np.array(probs)
    probs = (probs / probs.sum()).tolist()
    return probs, {}


def get_answer_memory_questions(story, question, choices, llm):
    probs = []
    state = get_initial_state_tomi(story, llm)
    state = "Initial state: " + state
    for c in choices:
        probs.append(
            get_likelihood_general(state, c, llm, variable="Memory", verbose=True)
        )
    probs = np.array(probs)
    probs = (probs / probs.sum()).tolist()
    return probs, {}


def split_sentences(story, llm):
    with open(
        f"prompts/prompts_{llm}/split_sentences.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Story]", f"Story: {story}")
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    return resp


if __name__ == "__main__":
    # get_context(
    #     "Aisha is a beekeeper in a small village near the Sahara Desert. The beehives appear to be full of honey and ready for harvest. Aisha is out of town and does not see the sand-filled beehives after the sandstorm the next day after she returns."
    # )
    # get_inf_var("If his goal was to hinder alice, what is his belief?")
    # get_info_from_question("Which one of the following statements is more likely to be true?")
    # get_variables(
    # update_state(
    #     "The peach is in the envelope.", "Amelia entered the garden.", "gpt-4o"
    # )
    #     story="""Angela has a phobia of bees, and she gets very nervous when there are bees flying around. Today, I gifted her a bottle of honey.""",
    # var_name_list=["Observation", "State", "Belief", "Goal", "Action", "Emotion"]
    # )

    # entire_story = "Imagine that you are in the perspective of David in a world where their action and potentially other people's actions in the physical state of the world are described as Context: What's inside the apartment: The apartment consists of a bedroom, kitchen, living room, and bathroom.  In the bedroom, there is a cabinet, coffee table, desk, and sofa. The cabinet houses a bag of chips, a wine glass, and two books, while the sofa holds a bag of chips and a book.  The kitchen is equipped with a dishwasher, eight cabinets, a kitchen table, a fridge, a microwave, and a stove. Inside the dishwasher, there are two wine glasses and a water glass. The first, second, third, fourth, and fifth cabinets from left to right are empty, while the sixth cabinet contains an apple. The seventh cabinet has a wine glass, and the eighth cabinet is also empty. The kitchen table is adorned with a book, a dish bowl, a plate, a condiment bottle, and a salmon. The fridge contains a bottle of wine, an apple, and a salmon. The microwave houses a condiment bottle, a cupcake, and a plate, and there is a plate inside the stove.  The living room features a coffee table, a sofa, and a desk. On the coffee table, there is a wine glass, a plate, and two remote controls, and a book is placed on the sofa.  The bathroom is fitted with a cabinet, which is currently empty.  Actions taken by David: David is situated in the bedroom. David proceeds towards the kitchen.David approaches the second kitchen cabinet.David opens the second kitchen cabinet.David closes the second kitchen cabinet.David opens the first kitchen cabinet.David closes the first kitchen cabinet.as well. David moves towards the dishwasher.David opens the dishwasher.David shuts the dishwasher.David opens the sixth kitchen cabinet.David closes the sixth kitchen cabinet.David walks towards the stove.David opens the stove.David closes the stove.David opens the fifth kitchen cabinet.David closes the fifth kitchen cabinet.David returns to the bedroom.David approaches a cabinet in the bedroom.David opens the cabinet in the bedroom.David closes the cabinet in the bedroom.David heads back to the kitchen.David walks towards the third kitchen cabinet.David opens the third kitchen cabinet.David closes the third kitchen cabinet.David proceeds towards the eighth kitchen cabinet. David has been trying to get a bottle of wine.NONE."
    # story = "What's inside the apartment: The apartment consists of a bedroom, kitchen, living room, and bathroom.  In the bedroom, there is a cabinet, coffee table, desk, and sofa. The cabinet houses a bag of chips, a wine glass, and two books, while the sofa holds a bag of chips and a book.  The kitchen is equipped with a dishwasher, eight cabinets, a kitchen table, a fridge, a microwave, and a stove. Inside the dishwasher, there are two wine glasses and a water glass. The first, second, third, fourth, and fifth cabinets from left to right are empty, while the sixth cabinet contains an apple. The seventh cabinet has a wine glass, and the eighth cabinet is also empty. The kitchen table is adorned with a book, a dish bowl, a plate, a condiment bottle, and a salmon. The fridge contains a bottle of wine, an apple, and a salmon. The microwave houses a condiment bottle, a cupcake, and a plate, and there is a plate inside the stove.  The living room features a coffee table, a sofa, and a desk. On the coffee table, there is a wine glass, a plate, and two remote controls, and a book is placed on the sofa.  The bathroom is fitted with a cabinet, which is currently empty.  Actions taken by David: The stove is closed."
    # character = "David"

    # entire_story = """ 1 Jack entered the bathroom.
    # 2 Logan entered the bathroom.
    # 3 Jack hates the asparagus.
    # 4 The celery is in the pantry.
    # 5 The pantry is in the bathroom.
    # 6 Olivia entered the bathroom.
    # 7 Logan moved the celery to the crate.
    # 8 The crate is in the bathroom.
    # 9 Jack exited the bathroom.
    # 10 Logan exited the bathroom.
    # 11 Jack hates the orange.
    # 12 Jack entered the bathroom. """
    # story = "Jack entered the bathroom.2 Logan entered the bathroom. 3 Jack hates the asparagus. 4 The celery is in the pantry. 5 The pantry is in the bathroom. 6 Olivia entered the bathroom. 7 Logan moved the celery to the crate. 8 The crate is in the bathroom. "
    # character = "Jack"

    # K = 2
    llm = "gpt-4o"
    # obs = hypothesis_generation_observation_inf_wrld_rules(
    #     entire_story, story, character, K, llm
    # )
    # print(obs)

    story = "Aniket is a marine biologist studying coral reefs off the coast of India. Aniket needs to collect samples of coral to analyze the effects of climate change on the reef. Aniket spots a healthy-looking coral formation in a specific area of the reef. A sudden wave surge stirs up sediment  covering the once healthy coral formation and causing it to become damaged. Aniket notices the wave surge and the sediment covering the coral."
    character = "Aniket"
    var_name_list = "Be"
    K = 2
    inf_var_name = "Belief"
    dataset_name = "BigToM"
    info = "Aniket sees that there is another healthy coral formation.', 'Aniket sees that he found a healthy coral formation.'"
    # "Aniket will find another healthy coral formation to collect samples.; Aniket will collect samples from the healthy coral formation he found."
    vars = hypothesis_generation(
        "", info, story, character, inf_var_name, K, llm, dataset_name
    )
    print(vars)
