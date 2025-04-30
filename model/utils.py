import json
from openai import OpenAI
import csv
import string
import math
import os
import base64
from PIL import Image
import pandas as pd
import io
import cv2
import transformers
import torch
import time
import glob


def get_info(s, num=0):
    i = num
    ret = []
    while True:
        i += 1
        pat = f"{i}."
        nxt = f"{i+1}."
        if pat in s:
            ret.append(s.split(pat)[1].split(nxt)[0].strip())
        else:
            break
    return ret


def extract_frames(video_id, num_frames=8):
    cap = cv2.VideoCapture(f"../benchmarks/data/MuMa/videos/video_{video_id}.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = total_frames // num_frames

    frame_indices = [i * interval for i in range(num_frames)]

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = f"../benchmarks/data/MuMa/frames/{video_id}_frame_{idx}.png"
            cv2.imwrite(output_path, frame)

    cap.release()


def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError:
        # Create an empty image and return its encoding if the file is not found
        empty_image = Image.new("RGB", (1, 1))
        buffered = io.BytesIO()
        empty_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


cost_of_information_extracting = 0.0
cost_of_proposing_hypotheses = 0.0
times_of_information_extracting = 0
times_of_proposing_hypotheses = 0


def gpt_request_multimodal(
    prompt,
    base64_images,
    temperature=0.0,
    max_tokens=3000,
    model="gpt-4o",
    hypo=False,
    verbose=False,
):
    global cost_of_information_extracting
    global cost_of_proposing_hypotheses
    global times_of_information_extracting
    global times_of_proposing_hypotheses
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        content_list = [
            {
                "type": "text",
                "text": prompt,
            }
        ]
        base64_images = base64_images[:32]
        for i in range(len(base64_images)):
            content_list.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_images[i]}",
                        "detail": "low",
                    },
                },
            )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content_list}],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=42,
        )
        if model == "gpt-4":
            inp, op = 30 / 1000000, 60 / 1000000
        elif model == "gpt-4o":
            inp, op = 5 / 1000000, 15 / 1000000
        elif model == "gpt-3.5-turbo":
            inp, op = 0.5 / 1000000, 1.5 / 1000000
        usage = response.usage
        cost = usage.prompt_tokens * inp + usage.completion_tokens * op
        if hypo:
            cost_of_proposing_hypotheses += cost
            times_of_proposing_hypotheses += 1
            if times_of_proposing_hypotheses % 10 == 0:
                enh_print(
                    f"Accumulated Cost of Proposing Hypotheses: {cost_of_proposing_hypotheses} in {times_of_proposing_hypotheses} times",
                    "red",
                )
        else:
            cost_of_information_extracting += cost
            times_of_information_extracting += 1
            if times_of_information_extracting % 10 == 0:
                enh_print(
                    f"Accumulated Cost of Extracting Information: {cost_of_information_extracting} in {times_of_information_extracting} times",
                    "red",
                )
        # if verbose:
        enh_print(prompt)
        enh_print(response.choices[0].message.content.strip(), color="red")
        enh_print(f"cost: {cost}", color="red")
        return response.choices[0].message.content.strip(), cost
    except Exception as e:
        print("An error occurred:", e)
        return "Error.", 0


def llm_request(
    prompt, temperature=0.0, max_tokens=3000, model="gpt-4o", hypo=False, verbose=False
):
    if "gpt" in model:
        return gpt_request(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model="gpt-4o",
            hypo=hypo,
            verbose=verbose,
        )
    elif "llama" in model:
        return llama_request(prompt, max_tokens=200)


def llama_request(prompt, max_tokens=200):
    API_TOKEN = os.environ["LLAMA_API_KEY"]
    cache_directory = "/scratch/tshu2/cjin33"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    if torch.cuda.is_available():
        device = 0  # Assuming you want to use the first GPU
        print("GPU Available: Using GPU")
    else:
        device = -1  # Use CPU
        print("GPU Not Available: Using CPU")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,  # "auto",
        token=API_TOKEN,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=3000,
    )
    generated_text = outputs[0]["generated_text"][-1]["content"]
    generated_text = generated_text.replace("\n", " ")

    print(f"Prompt: {prompt}\n\n")
    print(f"LLaMA answer length: {len(generated_text)}")
    print(f"LLaMA answer: {generated_text}")

    cost = 0

    return generated_text, cost


def gpt_request(
    prompt,
    temperature=0.0,
    max_tokens=3000,
    model="gpt-4o",
    hypo=False,
    verbose=False,
    message_role="user",
    seed=42,
    logprobs=False,
    top_p=0,
    top_logprobs=5,
    action_exponent=None,
    variable=None,
):
    global cost_of_information_extracting
    global cost_of_proposing_hypotheses
    global times_of_information_extracting
    global times_of_proposing_hypotheses
    
    
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    
    cache_file = os.path.join(cache_dir, "prompt_response_cache.json")
    
    cache_data = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except Exception as e:
            enh_print(f"Error reading cache: {e}", "red")
    
    cache_key = f"{prompt}_{model}_{temperature}_{max_tokens}_{seed}_{logprobs}"
    if logprobs:
        cache_key += f"_{top_p}_{top_logprobs}"
    if action_exponent is not None and variable is not None:
        cache_key += f"_{action_exponent}_{variable}"
        
    if cache_key in cache_data:
        enh_print(f"Using cached response for prompt: {prompt[:50]}...", "yellow")
        return cache_data[cache_key]["response"], cache_data[cache_key]["cost"]
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        # getting the liklihood with gpt
        if logprobs:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": message_role,
                        "content": prompt,
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                top_p=top_p,
                top_logprobs=top_logprobs,
                logprobs=logprobs,
            )

        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": message_role,
                        "content": prompt,
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
        if model == "gpt-4":
            inp, op = 30 / 1000000, 60 / 1000000
        elif model == "gpt-4o":
            inp, op = 5 / 1000000, 15 / 1000000
        elif model == "gpt-3.5-turbo":
            inp, op = 0.5 / 1000000, 1.5 / 1000000
        usage = response.usage
        cost = usage.prompt_tokens * inp + usage.completion_tokens * op
        if hypo:
            cost_of_proposing_hypotheses += cost
            times_of_proposing_hypotheses += 1
            if times_of_proposing_hypotheses % 10 == 0:
                enh_print(
                    f"Accumulated Cost of Proposing Hypotheses: {cost_of_proposing_hypotheses} in {times_of_proposing_hypotheses} times",
                    "red",
                )
        else:
            cost_of_information_extracting += cost
            times_of_information_extracting += 1
            if times_of_information_extracting % 10 == 0:
                enh_print(
                    f"Accumulated Cost of Extracting Information: {cost_of_information_extracting} in {times_of_information_extracting} times",
                    "red",
                )
        # if verbose:
        # enh_print(prompt)
        # enh_print(response.choices[0].message.content.strip(), color="red")
        # getting the liklihood
        result = None
        if logprobs:
            response_json_str = response.model_dump_json(indent=2)
            response_dict = json.loads(response_json_str)
            logprob_a = None
            if verbose:
                print(
                    response_dict["choices"][0]["logprobs"]["content"][0][
                        "top_logprobs"
                    ]
                )
            for logprob in response_dict["choices"][0]["logprobs"]["content"][0][
                "top_logprobs"
            ]:
                if str(logprob["bytes"]) == str([65]):
                    logprob_a = logprob["logprob"]
                    break
            if logprob_a == None:
                prob_a = None
            else:
                prob_a = math.exp(logprob_a)
            if prob_a == None:
                if verbose:
                    print(
                        f"Encountering None values in prob_a!!\n\n{prompt}\n\n{response_dict}"
                    )
                result = 0.1
            else:
                if prob_a < 0.03:
                    prob_a = 0.03
                if prob_a > 0.97:
                    prob_a = 1.0
                # clip the values
                if action_exponent is not None and "Action" in variable:
                    result = math.pow(prob_a, action_exponent)
                else:
                    result = prob_a
                if verbose:
                    print(prompt, "\n", prob_a)
        else:
            result = response.choices[0].message.content.strip()
        
        cache_data[cache_key] = {
            "response": result,
            "cost": cost,
            "timestamp": time.time()
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            enh_print(f"Error writing cache: {e}", "red")
        
        return result, cost
    except Exception as e:
        print(f"retrying due to an error {e}")
        time.sleep(20)
        return gpt_request(prompt, temperature, max_tokens, model, hypo, verbose)


def enh_print(x, color="green"):
    if color == "green":
        print(f"\033[92m{x}\033[0m")
    elif color == "red":
        print(f"\033[91m{x}\033[0m")
    elif color == "yellow":
        print(f"\033[93m{x}\033[0m")
    else:
        print(x)


def return_letters(n):
    alphabet = list(
        string.ascii_uppercase
    )  # Creates a list of uppercase letters ['A', 'B', 'C', ..., 'Z']
    return alphabet[:n]  # Returns the first n letters


accumulated_cost_logits = 0


def get_logits(info, question, choices, model="gpt-4o"):
    global accumulated_cost_logits
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    inst = f"""Answer the question based on the story.
Story: {info}
Question: {question}
"""
    letter_choices = return_letters(len(choices))
    format_choices = ""
    for i, c in enumerate(choices):
        format_choices += f"{letter_choices[i]}) {c}\n"
    prompt = f"{inst}{format_choices}Answer:"
    # print(f'\n\n{prompt}\n\n')
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
            seed=42,
        )
        resp_json = response.model_dump_json(indent=2)
        resp_dict = json.loads(resp_json)
        logprobs = resp_dict["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        logits = {}
        # print(logprobs)

        if model == "gpt-4":
            inp, op = 30 / 1000000, 60 / 1000000
        elif model == "gpt-4o":
            inp, op = 5 / 1000000, 15 / 1000000
        elif model == "gpt-3.5-turbo":
            inp, op = 0.5 / 1000000, 1.5 / 1000000
        usage = response.usage
        cost = usage.prompt_tokens * inp + usage.completion_tokens * op

        accumulated_cost_logits += cost
        print("accumulated cost: ", accumulated_cost_logits)

        for it in logprobs:
            for c in letter_choices:
                if it["token"] == c:
                    logits[c] = math.exp(it["logprob"])
        return prompt, logits
    except Exception as e:
        print(f"retrying due to an error {e}")
        time.sleep(20)
        return get_logits(info, question, choices, model)

def contains_utterance(self, data_list_1, data_list_2):
    if data_list_2 == None:
        return False
    for dictionary in data_list_2:
        for key, value in dictionary.items():
            if "Utterance" in key and value != "NONE":
                return True
    return False

def check_nested(self):
    # Safety check. If not satisfied, it is classified as incorrect.
    nested_dataset = ("2nd" in self.dataset_name)
    if "HiToM" in self.dataset_name:
        order = int(self.dataset_name.split('order')[1])
        if order > 1:
            nested_dataset = True
    if self.nested == True and not nested_dataset:
        return False
    return True

def letter_to_number_mapping(letter):
    return ord(letter.upper()) - ord("A")


def number_to_letter_mapping(number):
    return chr(number + ord("A"))


def parse_extraction(resp):
    if resp[0][0] == "A":
        return resp[1].strip().replace('"', "").strip()
    else:
        return "NONE"


def rephrase_choices_wording(c, story, llm):
    with open(
        f"prompts/prompts_{llm}/rephrase_choices_wording.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Sentence]", f"Sentence: {c}")
    prompt = prompt.replace("[Story]", f"Story: {story}")
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    resp = resp.replace("Output:", "").strip()
    return resp


def rephrase_choices(question, choices, llm, wording=False):
    with open(
        f"prompts/prompts_{llm}/rephrase_choices.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Question]", f"Question: {question}")
    prompt = prompt.replace("[Choices]", f"Choices: {choices}")
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    resp = resp.replace("'s", "s")
    
    resp = resp.strip()
    try:
        from ast import literal_eval
        return literal_eval(resp)
    except (ValueError, SyntaxError) as e:
        # If direct eval fails, try cleaning the string
        try:
            resp = resp.replace("'s", "s")
            resp = resp.replace("'t", "t")
            return literal_eval(resp)
        except (ValueError, SyntaxError) as e:
            print(f"Failed to parse response: {resp}")
            return []  # Return empty list as fallback


def find_inference_timestep(story, choices, llm):
    with open(
        f"prompts/prompts_{llm}/find_inference_timestep.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Story]", f"Story: {story}")
    prompt = prompt.replace("[Choice]", f"given sentence: {choices}")
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    resp = resp.split("\n")[0].strip()
    resp = resp.replace('"', "'")
    story = story.split(resp)[0] + resp
    # print(story)
    # quit()
    return story

def find_relevant_entities(choices, agents, llm):
    with open(
        f"prompts/prompts_{llm}/find_relevant_entities.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    
    entities = set()
    for c in choices:
        prompt = prompt_template.replace("[Choice]", f"{c}")
        resp, cost = llm_request(prompt, temperature=0.0, model=llm)
        resp_entities = get_list_from_str(resp)
        entities.update(resp_entities) 
    entities.update(agents)
    # print(prompt, '\n', entities)
    print('entities extracted: ', entities)
    return list(entities)


def rewrite_belief_info(info, init_states, llm):
    with open(
        f"prompts/prompts_{llm}/rewrite_belief_info.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    if "know" not in info:  # not belief sentence
        return info
    prompt = prompt_template.replace("[Sentence]", f"Sentence: {info}")
    prompt = prompt.replace("[States]", f"States: {init_states}")
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    resp = resp.split("\n")[0].strip()
    resp = resp.replace('"', "'")
    # print('rewrite belief info:', info, init_states, resp)
    # print(story)
    # quit()
    return resp

def find_nested_agent_list(question, choices, llm):
    # Just to get GT. Replace this with LLM-based extraction later.
    first_agent = question.split('think')[0].split('does')[1].strip()
    other_agents = 'think'.join(question.split('think')[1:]).split('thinks')[:-1]
    oa = []
    for a in other_agents:
        oa.append(a.strip())
        
    return [first_agent] + oa

def reconstruct_story_nested(story, agent, llm, dataset_name):
    parsed_story = story.split(".")
    ret = []
    vis = []
    for sentence in parsed_story:
        sentence = sentence.strip()
        if sentence == "":
            continue
        sentence = sentence + "."
        rec = ""
        if rephrase_story_nested_single(story, agent, sentence, llm, dataset_name):
            ret.append(sentence)
            rec = sentence
        vis.append({
            "Original story": sentence,
            "Reconstructed story": rec
        })
    return ret, vis

def rephrase_story_nested_single(story, agent, sentence, llm, dataset_name):
    with open(
        f"prompts/prompts_{llm}/rephrase_story_nested_single.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Story]", f"Story: {story}")
    prompt = prompt.replace("[Agent]", agent)
    prompt = prompt.replace("[Sentence]", f"{sentence}")
    if "HiToM" in dataset_name:
        if "enter" in sentence or "exit" in sentence: # To represent initial state in HiToM, and HiToM assume the order of agents leaving is known
            return True
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    if resp[0] == "A":
        return True
    elif resp[0] == "B":
        return False
    else:
        return True


def rephrase_question_nested(question, agent, llm, dataset_name):
    # This function gives ground truth rephrased question in ToMi-2nd and Hi-ToM.
    # For more open-ended scenarios, replace this function with LLMs.
    if "thinks" in question:
        obj_q = question.split('thinks')[-1]
    else:
        obj_q = ' the ' + question.split('the ')[-1] + 'is?'
    return f"Where does {agent} think{obj_q}"

def mmtom_modality_fusion(video_info, text_variables, agent_name):
    pass

def story_fusion(video_story, text_story, llm):
    with open(
        f"prompts/prompts_{llm}/story_fusion.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[video_story]", f"Video story: {video_story}")
    prompt = prompt.replace("[text_story]", f"Text story: {text_story}")
    resp, cost = llm_request(prompt, temperature=0.0, model=llm)
    resp = resp.split("\n")[0]
    resp = resp.replace('"', "'")
    if 'A' in resp:
        return text_story + ' ' + video_story
    else:
        return text_story

def correct_visual_actions(action, choices, llm):
# visual action might have errors. Fuse text to correct it
    with open(f'prompts/prompts_{llm}/correct_visual_info.txt', 'r', encoding="utf-8") as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Action]", f"Action: {action}")
    prompt = prompt.replace("[Info]", f"Information: {choices}")
    resp, cost = llm_request(prompt, temperature=0.0, hypo=True, model=llm)
    # print(prompt, resp)
    return resp

def get_rid_of_number_starts(story):
    # Getting rid of number starts, and underscores to obtain a more natural story illustration
    # print(story, f'{1} ')
    for i in range(30, 0, -1):
        story = story.replace(f"{i} ", "")
    story = story.replace("_", " ")
    story = story.replace("\n", " ")
    return story


def create_folder_if_not_there(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_model_name(model):
    model_name = ""
    for var in model:
        model_name += var[0].lower()
    return model_name

def get_filename_with_episode_name(episode_name, base_path="../results/middle", suffix="csv"):

    pattern = os.path.join(base_path, f"*_{episode_name}.{suffix}")
    
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        print(f"No files found matching episode_name: {episode_name}, {base_path}")
        return None
    
    file_path = matching_files[0]
    print(f"Reading file: {file_path}")
    
    return file_path

def find_agents(story, llm):
    with open(
        f"prompts/prompts_{llm}/get_agent_names.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Story]", f"Story: {story}")
    resp, cost = llm_request(prompt, temperature=0, model=llm)
    # print(resp)
    resp = eval(resp)
    # print(resp)
    return resp


def find_inferred_agent(question, choices, llm):
    with open(
        f"prompts/prompts_{llm}/get_inferred_agent.txt", "r", encoding="utf-8"
    ) as prompt_file:
        prompt_template = prompt_file.read().strip()
    prompt = prompt_template.replace("[Question]", f"Question: {question}")
    prompt = prompt.replace("[Choices]", f"Choices: {choices}")
    resp, cost = llm_request(prompt, temperature=0, model=llm)
    # print(prompt, resp)
    return resp


def get_list_from_str(resp):
    try:
        resp_list = eval(resp)
    except SyntaxError:
        # Manual parsing for malformed list strings
        def parse_list_string(s):
            s = s.strip('[]')  # Remove outer brackets
            items = []
            current_item = ''
            in_quotes = False
            quote_char = None
            
            for char in s:
                if char in ['"', "'"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                elif char == ',' and not in_quotes:
                    items.append(current_item.strip().strip('"\''))
                    current_item = ''
                    continue
                current_item += char
            
            if current_item:
                items.append(current_item.strip().strip('"\''))
            
            return [item for item in items if item]

        resp_list = parse_list_string(resp)
    return resp_list