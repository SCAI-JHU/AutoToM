from ElementExtractor import *
from utils import *
from BayesianInference import BayesianInferenceModel

all_time_goal_likelihood = None
init_belief_probs = [0.0, 0.0, 0.0]

def infer_belief_at_timestamp(
    self,
    time_variables,
    i,
    previous_belief,
    belief_name,
    variable_values_with_time,
    all_probs,
    no_observation_hypothesis,
    all_prob_estimations,
    previous_actions=None,
    rational_agent_statement=False,
    approximate=False
):
    # For all time stamps except for the last --> we infer belief with Bayesian Inference
    # print(no_observation_hypothesis)
    global init_belief_probs
    if isinstance(time_variables, list):
        var_i = time_variables[i]
    elif isinstance(time_variables, dict):  # variables at a specific timestep
        var_i = time_variables
    now_variables = []
    for key, item in var_i.items():
        if (
            key != belief_name
            and key != "All Actions"
            and key != "Ground Truth State"
        ):
            now_variables.append(item)
    if belief_name in var_i:
        now_variables.append(var_i[belief_name])
    if "BigToM" in self.dataset_name:
        context = self.story
    else:
        context = ""
    inference_model = BayesianInferenceModel(
        variables=now_variables,
        context=context,
        llm=self.llm,
        verbose=self.verbose,
        inf_agent=self.inf_agent_name,
        model_name=self.model_name,
        episode_name=self.episode_name,
        dataset_name=self.dataset_name,
        K=self.K,
        answer_choices=self.choices,
        world_rules=self.world_rules,
        all_prob_estimations=all_prob_estimations,
        no_observation_hypothesis=no_observation_hypothesis,
        reduce_hypotheses=self.reduce_hypotheses,
        previous_actions=previous_actions,
        rational_agent_statement=rational_agent_statement,
        approximate=approximate
    )

    try:
        results, all_prob_estimations, all_node_results, all_NLDs = inference_model.infer("Belief", self.model_name, self.episode_name, self.init_belief)
    except Exception as e:
        print(f"Exception {e}")
        return (Variable("Previous Belief", True, False, ["NONE"], np.ones(1)), all_prob_estimations, all_probs)
    
    init_belief_probs = results
    self.save_NLD_descriptions(self, i, all_NLDs)
    self.translate_and_add_node_results(self, i, all_node_results)
    previous_belief = deepcopy(var_i[belief_name])
    previous_belief.prior_probs = np.array(results)
    previous_belief.name = "Previous Belief"
    # if self.verbose:
    enh_print(
        f"After time {i}: Belief Probs Updated to {previous_belief.possible_values}, {results}"
    )
    chunk = "NONE"
    if variable_values_with_time is None:
        chunk = "NONE"
    elif i < len(variable_values_with_time) and "Chunk" in variable_values_with_time[i]:
        chunk = variable_values_with_time[i]["Chunk"]
    if "Belief" in self.inf_var_name:
        now_probs = {
            "Time": i,
            "Chunk": chunk,
            f"Probs({self.choices})": results,
        }
        all_probs.append(now_probs)
    return (previous_belief, all_prob_estimations, all_probs)
    # return previous_belief, all_prob_estimations, all_probs

def infer_last_timestamp(
    self,
    time_variables,
    i,
    inf_name,
    inf_var_name,
    now_variables,
    no_observation_hypothesis,
    variable_values_with_time,
    all_probs,
    all_prob_estimations,
    action_likelihood_goal,
    previous_actions=None,
    rational_agent_statement=False,
    approximate=False
):
    
    # print(previous_actions)
    # Last time stamp --> we want to infer the variable we are interested in with Bayesian Inference
    if isinstance(time_variables, list):
        var_i = time_variables[i]
    elif isinstance(time_variables, dict):  # variables at a specific timestep
        var_i = time_variables
    for key, item in var_i.items():
        if key != inf_name and key != "All Actions" and key != "Ground Truth State":
            if item != "NONE":
                item.name = key
                now_variables.append(item)

    try:
        now_variables.append(var_i[inf_name])
    except Exception:
        print(f"No {inf_name}!")
        return None

    if self.verbose:
        print("chosen variables: \n\n\n", now_variables, "\n\n\n")

    if "BigToM" in self.dataset_name:
        context = self.story
    else:
        context = "" 
    inference_model = BayesianInferenceModel(
        variables=now_variables,
        context=context,
        llm=self.llm,
        verbose=self.verbose,
        inf_agent=self.inf_agent_name,
        model_name=self.model_name,
        episode_name=self.episode_name,
        dataset_name=self.dataset_name,
        K=self.K,
        answer_choices=self.choices,
        world_rules=self.world_rules,
        all_prob_estimations=all_prob_estimations,
        no_observation_hypothesis=no_observation_hypothesis,
        reduce_hypotheses=self.reduce_hypotheses,
        previous_actions=previous_actions,
        rational_agent_statement=rational_agent_statement,
        approximate=approximate
    )

    results, all_prob_estimations, all_node_results, all_NLDs = inference_model.infer(inf_var_name, self.model_name, self.episode_name, self.init_belief)
    self.translate_and_add_node_results(self, i, all_node_results)
    self.save_NLD_descriptions(self, i, all_NLDs)
    save_node_results(
        self.intermediate_node_results,
        self.model_name,
        self.episode_name,
        self.back_inference,
        self.reduce_hypotheses,
    )
    if inf_var_name == "Goal":
        action_likelihood_goal[i] = results
        # print(results)
        # print(inf_name, var_i[inf_name].possible_values)
        accumulative_results = np.ones((len(var_i[inf_name].possible_values)))
        accumulative_results /= accumulative_results.sum()
        # print(accumulative_results, action_likelihood_goal)
        for k, v in action_likelihood_goal.items():
            accumulative_results *= np.array(v)
            accumulative_results /= accumulative_results.sum()
        enh_print(
            f"After time {i}: {inf_name} Probs calculated as {var_i[inf_name].possible_values}, {accumulative_results}"
        )
        enh_print(
            f"Goal probs at different time steps: {action_likelihood_goal}"
        )
        global all_time_goal_likelihood
        all_time_goal_likelihood = deepcopy(action_likelihood_goal)
        results = accumulative_results
    else:
        enh_print(
            f"After time {i}: {inf_name} Probs calculated as {var_i[inf_name].possible_values}, {results}"
        )
    chunk = "NONE"
    if variable_values_with_time is None:
        chunk = "NONE"
    elif i < len(variable_values_with_time) and "Chunk" in variable_values_with_time[i]:
        chunk = variable_values_with_time[i]["Chunk"]
    now_probs = {
        "Time": i,
        "Chunk": chunk,
        f"Probs({self.choices})": results,
    }
    all_probs.append(now_probs)
    return results, all_prob_estimations, all_probs


def infer_goal_at_timestamp(
    self,
    time_variables,
    i,
    previous_belief,
    belief_name,
    goal_name,
    variable_values_with_time,
    all_probs,
    no_observation_hypothesis,
    all_prob_estimations,
    previous_actions=None,
    rational_agent_statement=False,
    approximate=False
):
    # If we're inferring goal, we need to record P(Action | Goal, ...) at every timestep (we assume the agent has a consistent goal)
    # Same with belief, we infer goal with Bayesian Inference. But notice that the compute (API calls / tokens) will not increase, because the likelihoods needed are already stored in the cache.

    if isinstance(time_variables, list):
        var_i = time_variables[i]
    elif isinstance(time_variables, dict):  # variables at a specific timestep
        var_i = time_variables
    now_variables = []
    for key, item in var_i.items():
        if (
            key != goal_name
            and key != "All Actions"
            and key != "Ground Truth State"
        ):
            now_variables.append(item)
    if goal_name in var_i:
        now_variables.append(var_i[goal_name])
    if "BigToM" in self.dataset_name:
        context = self.story
    else:
        context = ""
    inference_model = BayesianInferenceModel(
        variables=now_variables,
        context=context,
        llm=self.llm,
        verbose=self.verbose,
        inf_agent=self.inf_agent_name,
        model_name=self.model_name,
        episode_name=self.episode_name,
        dataset_name=self.dataset_name,
        K=self.K,
        answer_choices=self.choices,
        world_rules=self.world_rules,
        all_prob_estimations=all_prob_estimations,
        no_observation_hypothesis=no_observation_hypothesis,
        reduce_hypotheses=self.reduce_hypotheses,
        previous_actions=previous_actions,
        rational_agent_statement=rational_agent_statement,
        approximate=approximate
    )

    try:
        results, all_prob_estimations, all_node_results, all_NLDs = inference_model.infer("Goal", self.model_name, self.episode_name, self.init_belief)
    except Exception as e:
        print(f"Exception {e}")
        return [1.0 for _ in range(len(var_i[goal_name].possible_values))], all_prob_estimations, all_probs
    self.save_NLD_descriptions(self, i, all_NLDs)
    self.translate_and_add_node_results(self, i, all_node_results)
    # if self.verbose:
    enh_print(
        f"At time {i}: Goal Probs calculated as {var_i[goal_name].possible_values}, {results}"
    )
    chunk = "NONE"
    if variable_values_with_time is None:
        chunk = "NONE"
    elif i < len(variable_values_with_time) and "Chunk" in variable_values_with_time[i]:
        chunk = variable_values_with_time[i]["Chunk"]

    now_probs = {
        "Time": i,
        "Chunk": chunk,
        f"Probs({self.choices})": results,
    }
    all_probs.append(now_probs)
    return (results, all_prob_estimations, all_probs)
    # return previous_belief, all_prob_estimations, all_probs
