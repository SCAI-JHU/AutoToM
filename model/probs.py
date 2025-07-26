from openai import OpenAI
import json
import math
import string
import numpy as np
from utils import *
import openai
import os

cost_of_estimating_likelihood = 0.0
times_of_estimating = 0


def return_letters(n):
    alphabet = list(
        string.ascii_uppercase
    )  # Creates a list of uppercase letters ['A', 'B', 'C', ..., 'Z']
    return alphabet[:n]  # Returns the first n letters


def get_likelihood(
    info,
    statement,
    dataset_name,
    model="gpt-4o-2024-11-20",
    verbose=False,
    world_rules=None,
    variable=None,
    inf_agent=None,
    action_exponent=None,
    rational_agent_statement=False,
):
    likelihood = get_likelihood_general(
        info,
        statement,
        model,
        verbose,
        world_rules,
        variable,
        inf_agent,
        action_exponent,
        rational_agent_statement=rational_agent_statement
    )
    return likelihood


def get_likelihood_general(
    info,
    statement,
    model="gpt-4o-2024-11-20",
    verbose=False,
    world_rules=None,
    variable=None,
    inf_agent=None,
    action_exponent=None,
    rational_agent_statement=False
):

    if "Observation" in variable:
        prompt = f"""Determine if the statement is likely, respond with only either A or B.
{info}
Here is a statement of {inf_agent}'s current observation. Only evaluate current observation of {inf_agent} based on the state. Do not imagine anything else. Think about {inf_agent}'s location. {inf_agent} is quite likely to observe all objects and events in {inf_agent}'s location, and is unlikely to observe states in another location. If {inf_agent} does not appear in the state, {inf_agent} can't observe anything. Note that the statement has to be precise in wording to be likely. For example, treasure chest and container are different in wording and they're different objects.
Determine if the following statement is likely: {statement}
A) Likely.
B) Unlikely."""
    elif "Initial Belief" in variable:
        prompt = f"""Determine if the statement is likely, respond with only either A or B. If it's not certain but it's possible, it's considered likely.
    Here is a statement of the story and {inf_agent}' initial belief. 
    
    There is an action that causes the state of the main object to change. Based on {inf_agent}'s observations determine if {inf_agent} perceives the state of the object change. 
    If it is not clearly stated that {inf_agent} perceives it then we do not assume that {inf_agent} perceived the change of state. 
    If {inf_agent} perceives this change then it is highly likely that {inf_agent}'s belief aligns with the change of state of the object. 
    If {inf_agent} does not perceive this change or if it is unknown if {inf_agent} perceives this change then it is highly likely that {inf_agent}'s belief does not align with the change of state of the object. 

    
    Story: {info}
    Think about the state of the world and others actions. {inf_agent}' belief can change throughout time through other's actions and what {inf_agent} can observe. It is also important to think about if {inf_agent} can observe other's actions. If {inf_agent} can observe the same then their belief will change and if not then their belief will remain constant. Use this to determine {inf_agent}'s beliefs.
    Determine if the following statement is likely: {statement}
    A) Likely.
    B) Unlikely."""
    elif "Actions" in variable:
        actions = info[1]
        info = info[0]
        action_a = actions[0]
        action_b = actions[1]

        if action_a in statement:
            prompt = f"""Determine if the statement is likely, respond with only either A or B. If it's not certain but it's possible, it's likely.
                {info}
                If the next immediate actions possible are: {actions}
                Determine which immediate action is most possible given the about {inf_agent}'s goal and belief.
            
                Determine if the following statement is likely: {action_a} is a better immediate action than {action_b}. 
                A) Likely.
                B) Unlikely."""
        else:
            prompt = f"""Determine if the statement is likely, respond with only either A or B. If it's not certain but it's possible, it's likely.
                {info}
                If the next immediate actions possible are: {actions}
                Determine which immediate action is most possible given the about {inf_agent}'s goal and belief.
            
                Determine if the following statement is likely: {action_b} is a better immediate action than {action_a}. 
                A) Likely.
                B) Unlikely."""
        # print(prompt)

    elif "Action" in variable:
        if "Belief of Goal" in info:  # P(Action | Goal, Belief, Belief of Goal)
            prompt = f"""Determine if {inf_agent}'s action is likely, respond with only either A or B.
{info}
{inf_agent}'s action: {statement}
When {inf_agent} wants to help, {inf_agent} is likely to bring an object to other's desired location, and unlikely to grab an object away from other's desired location.
When {inf_agent} wants to hinder, {inf_agent} is likely to grab an object away from other's desired location, and unlikely to bring an object to other's desired location.
When {inf_agent} doesn't know other's goal, {inf_agent} is likely to act according to {inf_agent}'s belief.
If {inf_agent} wants to help and {inf_agent} believed the object is placed at other's desired location, it's unlikely {inf_agent} will move the object.
If {inf_agent}'s goal, {inf_agent}'s belief of goal, and {inf_agent}'s action do not align in any way, the action is unlikely.
Determine if {inf_agent}'s action is likely.
A) Likely.
B) Unlikely."""
        elif "Social Goal" in info:  # P(Action | Social Goal, Belief) in multi-agent setting
            prompt = f"""Determine if the statement is likely, respond with only either A or B. If it's not certain but it's possible, it's likely.
{info}
Here is a statement of {inf_agent}'s action. Think about {inf_agent}'s goal.
{inf_agent} will perform actions according to {inf_agent}'s belief, and any action that does not align with the belief is very unlikely, except when {inf_agent}'s goal is to hinder or to prevent others, and in this case (goal is hindering others) {inf_agent}'s action is only likely when it's different with {inf_agent}'s belief. If {inf_agent}'s mental states contains conditions like "When giving information" and the action is not giving information, it's unlikely.
Determine if the following statement is likely: {statement}
A) Likely.
B) Unlikely."""
        else: # P(Action | Goal, Belief)
            if rational_agent_statement:
                rational_statement = f" {inf_agent} is very sure of their belief and will not perform unnecessary actions."
            else:
                rational_statement = ""
            prompt = f"""Determine if the statement is likely, respond with only either A or B.
{info}
Here is a statement of {inf_agent}'s action. The belief stands for {inf_agent}'s current belief which is true. {inf_agent} is likely to act according to goal and belief concerning certain objects (the wording for objects must be same. You should ignore the correlation of different objects. e.g., plate and apple are two different objects.) Notice that {inf_agent}'s belief does not represent the goal.
When belief and goal are irrelevant, and action is directly driven by goal, it's likely. When belief and goal are relevant (about exactly the same object) and they contradict (or irrelevant) with action, it's unlikely.{rational_statement}
Determine if the following statement is likely: {statement}
A) Likely.
B) Unlikely."""
    elif "Belief" in variable:
        if "Observation" in info:  # P(Belief | Observation, Previous Belief)
            prompt = f"""Determine if the statement is likely, respond with only either A or B.
{info}
Here is a statement of {inf_agent}'s current belief. If {inf_agent}'s current belief is not aligned with {inf_agent}'s observation, it is very unlikely.
Determine if the following statement is likely: {statement}
A) Likely.
B) Unlikely."""
        else:  # P(Belief | State, Previous Belief)
            prompt = f"""Determine if the statement is likely, respond with only either A or B.
{info}
Here is a statement of {inf_agent}'s current belief. If {inf_agent}'s current belief is not aligned with the state, it is very unlikely.
Determine if the following statement is likely: {statement}
A) Likely.
B) Unlikely."""
    elif "Utterance" in variable:
        prompt = f"""Determine if {inf_agent}'s utterance is likely, respond with only either A or B.
{info}
{inf_agent}'s utterance: {statement}
When {inf_agent}'s goal is to help others, {inf_agent}'s utterance is likely when it strictly reflect {inf_agent}'s belief, and unlikely if it does not reflect {inf_agent}'s belief.
When {inf_agent}'s goal is to hinder or to prevent others from achieving their goals, {inf_agent}'s utterance is likely when it's different from {inf_agent}'s belief, and unlikely if it reflects {inf_agent}'s belief.
Determine if {inf_agent}'s utterance is likely.
A) Likely.
B) Unlikely."""
    else:
        prompt = f"""Determine if the statement is likely, respond with only either A or B.  If it's not certain but it's possible, it's considered likely. If it contradicts to the given information in some way, then it is unlikely. 
{info}
Determine if the following statement is likely: {statement}
A) Likely.
B) Unlikely."""
    if "gpt" in model:
        global cost_of_estimating_likelihood
        global times_of_estimating
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
            ],
            model=model,
            logprobs=True,
            top_p=0,
            top_logprobs=5,
            temperature=0.0,
            seed=0,
            max_tokens=1,
        )
        if model == "gpt-4":
            inp, op = 30 / 1000000, 60 / 1000000
        elif "gpt-4o" in model:
            inp, op = 5 / 1000000, 15 / 1000000
        elif model == "gpt-3.5-turbo":
            inp, op = 0.5 / 1000000, 1.5 / 1000000

        usage = response.usage
        cost = usage.prompt_tokens * inp + usage.completion_tokens * op
        cost_of_estimating_likelihood += cost
        times_of_estimating += 1
        if times_of_estimating % 100 == 0:
            enh_print(
                f"Accumulated Cost of Estimating Likelihood: {cost_of_estimating_likelihood} in {times_of_estimating} times",
                "red",
            )

        response_json_str = response.model_dump_json(indent=2)
        response_dict = json.loads(response_json_str)
        logprob_a = None
        if verbose:
            print(response_dict["choices"][0]["logprobs"]["content"][0]["top_logprobs"])
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
            return 0.1

        # clip the values
        if action_exponent is not None and "Action" in variable:
            return math.pow(prob_a, action_exponent)
        # if verbose:
        print(prompt, "\n", prob_a)
        return prob_a


def get_likelihood_test(prompt, verbose=True):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
        ],
        model="gpt-4o",
        logprobs=True,
        top_p=0,
        top_logprobs=5,
        temperature=0.0,
        seed=0,
        max_tokens=1,
    )

    response_json_str = response.model_dump_json(indent=2)
    response_dict = json.loads(response_json_str)
    logprob_a = None
    if verbose:
        print(response_dict["choices"][0]["logprobs"]["content"][0]["top_logprobs"])
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
        print(f"Encountering None values in prob_a!!\n\n{prompt}\n\n{response_dict}")
        return 0.01
    if prob_a < 0.01:
        prob_a = 0.01
    # clip the values
    if verbose:
        print(prompt, "\n", prob_a)
    return prob_a
