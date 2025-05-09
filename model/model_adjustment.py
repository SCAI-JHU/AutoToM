import pandas as pd
from scipy.stats import entropy
import openai
import copy

benefit_threshold = 0.02
# terminate_threshold = 0.6
utility_terminate_threshold = -0.673

model_space = [
    ['State', 'Observation', 'Belief', 'Action', 'Goal'],   # POMDP
    ['State', 'Observation', 'Belief', 'Action'],           # POMDP variant without Goal
    ['State', 'Observation', 'Belief'],                     # Simple Markov Model
    ['State', 'Belief', 'Action', 'Goal'],                  # POMDP variant without Observation
    ['State', 'Belief', 'Action'],                          # POMDP variant without Observation and Goal
    ['State', 'Belief'],                                    # Simple Markov Model variant without Observation
    ['State', 'Action', 'Goal'],                            # MDP
    ['State', 'Action'],                                    # MDP variant without Goal
    ['State'],                                              # MDP variant without Action and Goal
]

all_variables = ["State", "Observation", "Belief", "Action", "Goal", "Utterance", "Belief of Goal"]

def model_discovery(start_timestep, all_timesteps, verbose, time_variables, previous_belief, inf_agent_name, inf_var_name, estimation_dictionary, \
                        infer_last_timestamp, no_observation_hypothesis, variable_values_with_time, all_probs, answerfunc, argmax, argmin, save_belief_probs, model_name, \
                        episode_name, infer_belief_at_timestamp, belief_name, get_variables_at_time, mmtom_get_variables_at_time, choices, K, llm, hypo_llm, hypo_method, full, \
                        preproposed_ob_hypos, last_state, inf_agent_action, assigned_models, file_path, clear_current_nodes, dataset_name, states, actions, question, precomputed_states, \
                        model_variables, no_model_adjustment, self, action_likelihood_goal):
    
    results, terminate, model_variables = Bayesian_inference(start_timestep, all_timesteps, verbose, time_variables, previous_belief, inf_agent_name, inf_var_name, estimation_dictionary, \
                        infer_last_timestamp, no_observation_hypothesis, variable_values_with_time, all_probs, answerfunc, argmax, argmin, save_belief_probs, model_name, \
                        episode_name, infer_belief_at_timestamp, belief_name, get_variables_at_time, mmtom_get_variables_at_time, choices, K, llm, hypo_llm, hypo_method, full, \
                        preproposed_ob_hypos, last_state, inf_agent_action, assigned_models, dataset_name, states, actions, question, precomputed_states, model_variables, self, action_likelihood_goal)
    
    print("Initial Terminate: ", terminate)

    if no_model_adjustment:
        return results, terminate, assigned_models, model_variables

    if terminate:
        return results, terminate, assigned_models, model_variables
    
    initial_utility = - entropy(results)
    print("Initial Utility: ", initial_utility)

    def model_experiment(assigned_models, model, model_variables):
        assigned_models_test = copy.deepcopy(assigned_models)
        assigned_models_test[start_timestep] = model 

        clear_current_hypotheses(file_path, start_timestep)
        clear_current_nodes(self, start_timestep)
        results, terminate, model_variables = Bayesian_inference(start_timestep, all_timesteps, verbose, time_variables, previous_belief, inf_agent_name, inf_var_name, estimation_dictionary, \
                        infer_last_timestamp, no_observation_hypothesis, variable_values_with_time, all_probs, answerfunc, argmax, argmin, save_belief_probs, model_name, \
                        episode_name, infer_belief_at_timestamp, belief_name, get_variables_at_time, mmtom_get_variables_at_time, choices, K, llm, hypo_llm, hypo_method, full, \
                        preproposed_ob_hypos, last_state, inf_agent_action, assigned_models_test, dataset_name, states, actions, question, precomputed_states, model_variables, self, action_likelihood_goal)
        print("Model test: ", model)
        print("Model test results: ", results)
        # utility = max(results)
        # utility = 1 / entropy(results)
        utility = - entropy(results)
        return utility
    

    recompute = False
    while assigned_models[start_timestep] != model_space[0]:
        variables_for_experiments = [
            var for var in ["Belief", "Observation", "Goal", "Action"]
            if var not in assigned_models[start_timestep]
        ]
        utility_experiments = {}
        for var in variables_for_experiments:
            model = modify_variables(assigned_models[start_timestep], [var])
            if model not in model_space:
                continue

            utility = model_experiment(assigned_models, model, model_variables)
            utility_experiments[var] = utility
        
        print("Initial Utility: ", initial_utility)
        print("Utility Experiments: ", utility_experiments)

        if utility_experiments == {}:
            break
        elif len(utility_experiments) > 1:
            recompute = True
            
        best_var, best_utility = max(utility_experiments.items(), key=lambda item: item[1])
        # if best_utility > terminate_threshold:
        if best_utility > utility_terminate_threshold:
            assigned_models[start_timestep] = modify_variables(assigned_models[start_timestep], [best_var])

            clear_current_hypotheses(file_path, start_timestep)
            clear_current_nodes(self, start_timestep)
            results, terminate, model_variables = Bayesian_inference(start_timestep, all_timesteps, verbose, time_variables, previous_belief, inf_agent_name, inf_var_name, estimation_dictionary, \
                        infer_last_timestamp, no_observation_hypothesis, variable_values_with_time, all_probs, answerfunc, argmax, argmin, save_belief_probs, model_name, \
                        episode_name, infer_belief_at_timestamp, belief_name, get_variables_at_time, mmtom_get_variables_at_time, choices, K, llm, hypo_llm, hypo_method, full, \
                        preproposed_ob_hypos, last_state, inf_agent_action, assigned_models, dataset_name, states, actions, question, precomputed_states, model_variables, self, action_likelihood_goal)
            return results, terminate, assigned_models, model_variables
        elif best_utility - initial_utility < benefit_threshold:
            break
        else:
            assigned_models[start_timestep] = modify_variables(assigned_models[start_timestep], [best_var])
            initial_utility = best_utility

    if recompute:
        clear_current_hypotheses(file_path, start_timestep)
        clear_current_nodes(self, start_timestep)
        results, terminate, model_variables = Bayesian_inference(start_timestep, all_timesteps, verbose, time_variables, previous_belief, inf_agent_name, inf_var_name, estimation_dictionary, \
                        infer_last_timestamp, no_observation_hypothesis, variable_values_with_time, all_probs, answerfunc, argmax, argmin, save_belief_probs, model_name, \
                        episode_name, infer_belief_at_timestamp, belief_name, get_variables_at_time, mmtom_get_variables_at_time, choices, K, llm, hypo_llm, hypo_method, full, \
                        preproposed_ob_hypos, last_state, inf_agent_action, assigned_models, dataset_name, states, actions, question, precomputed_states, model_variables, self)

    return results, terminate, assigned_models, model_variables
    


def Bayesian_inference(start_timestep, all_timesteps, verbose, time_variables, previous_belief, inf_agent_name, inf_var_name, estimation_dictionary, \
                        infer_last_timestamp, no_observation_hypothesis, variable_values_with_time, all_probs, answerfunc, argmax, argmin, save_belief_probs, model_name, \
                        episode_name, infer_belief_at_timestamp, belief_name, get_variables_at_time, mmtom_get_variables_at_time, choices, K, llm, hypo_llm, hypo_method, full, \
                        preproposed_ob_hypos, last_state, inf_agent_action, assigned_models, dataset_name, states, actions, question, precomputed_states, model_variables, self, action_likelihood_goal):
    
    verbose = False
    no_observation_hypothesis = "NONE"                        
    for i in range(start_timestep, all_timesteps):
        if verbose:
            print(f"------------- time stamp {i} -------------")
        time_variables[i]["Previous Belief"] = previous_belief

        if verbose:
            print("Time Variables: ", time_variables)
        now_variables = []
        inf_name = f"{inf_agent_name}'s {inf_var_name}"

        variable_types = []
        for var_name in assigned_models[i]:
            variable_types.append((inf_agent_name, var_name))
        if verbose:
            print("Variable Types: ", variable_types)

        if verbose:
            print("i: ", i)
            print("variable_values_with_time", variable_values_with_time)

        variable_types.append(("", "All Actions"))
        if i == start_timestep: # propose hypotheses for now model
            if "MMToM" not in dataset_name:
                now_time_variables, preproposed_ob_hypos, last_state, inf_agent_action = get_variables_at_time(
                    time=i, vals=variable_values_with_time[i], variable_types=variable_types,
                    inf_agent_name=inf_agent_name, inf_var_name=inf_var_name, choices=choices,
                    K=K, llm=llm, hypo_llm=hypo_llm, verbose=verbose,
                    hypo_method=hypo_method, full=full, preproposed_ob_hypos=preproposed_ob_hypos,
                    last_state=last_state, inf_agent_action=inf_agent_action, dataset_name=dataset_name, precomputed_states=precomputed_states
                )
            else:
                now_time_variables, preproposed_ob_hypos = mmtom_get_variables_at_time(
                    time=i, variable_types=variable_types,
                    inf_agent_name=inf_agent_name, inf_var_name=inf_var_name, choices=choices,
                    K=K, llm=llm, hypo_llm=hypo_llm, verbose=verbose,
                    hypo_method=hypo_method, full=full, preproposed_ob_hypos=preproposed_ob_hypos,
                    states=states, actions=actions, question=question
                )
            model_variables[i] = now_time_variables
            # print(model_variables)
        else: # other timestamp we use the previous model
            now_time_variables = model_variables[i]

        if len(preproposed_ob_hypos) > 0 and no_observation_hypothesis == "NONE":
            no_observation_hypothesis = preproposed_ob_hypos[-1]

        now_time_variables["Previous Belief"] = previous_belief
        variables = now_time_variables
        now_variables = []
        if verbose:
            print("Variables: ", variables)
        
        if i == all_timesteps - 1:
            
            results, estimation_dictionary, all_probs = infer_last_timestamp(
                self,
                i=i,
                time_variables=variables,
                inf_name=inf_name,
                inf_var_name=inf_var_name,
                now_variables=now_variables,
                no_observation_hypothesis=no_observation_hypothesis,
                variable_values_with_time=variable_values_with_time,
                all_probs=all_probs,
                all_prob_estimations=estimation_dictionary,
                action_likelihood_goal=action_likelihood_goal,
            )
            
            # determine if we can stop inference
            results = list(results)
            terminate = False

            # utility = 1 / entropy(results)
            utility = - entropy(results)
            if len(results) == 2:
                # if max(results) > terminate_threshold:
                if utility > utility_terminate_threshold:
                    terminate = True
            else:
                # if answerfunc == argmax and max(results) > terminate_threshold:
                if answerfunc == argmax and utility > utility_terminate_threshold:
                    terminate = True
                elif answerfunc == argmin and min(results) < 0.2:
                    terminate = True
            
            if terminate is True:
                save_belief_probs(all_probs, model_name, episode_name)
                # save_all_estimations(all_prob_estimations, self.episode_name)
            
            return results, terminate, model_variables

        else:
            if self.inf_var_name == "Goal":
                action_likelihood_goal[i], self.estimation_dictionary, all_probs = (
                    self.infer_goal_at_timestamp(
                        self,
                        time_variables=variables,
                        i=i,
                        previous_belief=previous_belief,
                        belief_name=belief_name,
                        variable_values_with_time=variable_values_with_time,
                        all_probs=all_probs,
                        no_observation_hypothesis=no_observation_hypothesis,
                        all_prob_estimations=self.estimation_dictionary,
                        goal_name=f"{self.inf_agent_name}'s Goal",
                    )
                )
            previous_belief, estimation_dictionary, all_probs = infer_belief_at_timestamp(
                self,
                i=i,
                time_variables=variables,
                previous_belief=previous_belief, 
                belief_name=belief_name, 
                variable_values_with_time=variable_values_with_time,
                all_probs=all_probs, 
                no_observation_hypothesis=no_observation_hypothesis, 
                all_prob_estimations=estimation_dictionary)
    


def gpt_call(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def initial_model_proposal(question, inference_var, nested, contains_utterance):
    example_question = "Sally has a ball. She puts it in her basket. When Sally goes out for a walk, Anne moves the ball out of the basket and puts it in the box. Where will Sally look for the ball?"
    example_answer = "['State', 'Observation', 'Belief']"
    prompt = f"What variables are necessary to solve this question? Please provide the answer without an explanation. \
        Please select from the following: ['State', 'Observation', 'Belief', 'Action', 'Goal'] \
        State: The true condition of the environment. This should always be included. \
        Observation: The observed information about the state. Include this when the agent has partial observations of the state. \
        Belief: The agent's current estimation of the true state based on the state or its observation. \
        Action: A move made by the agent, informed by the state or belief. Include this only when it is directly relevant to answering the question. \
        Goal: The objective the agent is trying to achieve. Include this only if 'Action' is included. \n\
        Question: {example_question} \n\
        Variables: {example_answer} \n\
        Question: {question} \n\
        Variables: "
    response = gpt_call(prompt)
    assigned_model = eval(response)
    
    if inference_var not in assigned_model:
        assigned_model.append(inference_var)
        assigned_model = sorted(assigned_model, key=all_variables.index)
    
    if assigned_model not in model_space:
        if "State" not in assigned_model:
            assigned_model.append("State")
        if "Observation" in assigned_model and "Belief" not in assigned_model:
            assigned_model.append("Belief")
        if "Goal" in assigned_model and "Action" not in assigned_model:
            assigned_model.append("Action")
        assigned_model = sorted(assigned_model, key=all_variables.index)
    
    # assert assigned_model in model_space
    temp_assigned_model = [model for model in assigned_model if model != "Belief of Goal"]
    assert temp_assigned_model in model_space

    if contains_utterance:
        assigned_model.append("Utterance")

    print("Initial Model proposed: ", assigned_model)
    # assigned_model = ['State', 'Observation', 'Belief', 'Action', 'Goal']
    return assigned_model

def determine_realistic_questions(question):
    prompt = f"Determine whether the question is about realistic physical states. \
        It is about physical world states if it involves no agent. \n\
        Please respond with 'Yes' or 'No'. \n\
        If the answer is 'Yes', the question often ends with 'Where is A really?'. Otherwise, respond 'No'. \n\
        Question: Where does Emily think Jack will look for the celery? \
        Realistic: No \n\
        Question: Where will Jack look for the celery? \
        Realistic: No \n\
        Question: Where was the celery initially? \
        Realistic: No \n\
        Question: Where is the celery really? \
        Realistic: Yes \n\
        Question: {question} \
        Realistic: "
    response = gpt_call(prompt)
    if response == "Yes":
        return True
    else:
        return False
    
def determine_memory_questions(question):
    prompt = f"Determine whether the question is about where an object was initially. \
        Please respond with 'Yes' or 'No'. \n\
        If the answer is 'Yes', the question often ends with 'Where was A initially?'. Otherwise, respond 'No'. \n\
        Question: Where does Emily think Jack will look for the celery? \
        Response: No \n\
        Question: Where will Jack look for the celery? \
        Response: No \n\
        Question: Where is the celery really? \
        Response: No \n\
        Question: Where was the celery initially? \
        Response: Yes \n\
        Question: {question} \
        Response: "
    response = gpt_call(prompt)
    if response == "Yes":
        return True
    else:
        return False


def determine_higher_order_belief(question):
    prompt = f"Determine whether the question is about a higher-order belief. \
        A higher-order belief refers to a belief about another person's belief, goal or action. \n\
        It is not a high-order belief if it only asks about one agent's belief. \n\
        Please respond with 'Yes' or 'No'. \n\
        If the answer is 'Yes', the question often ends with 'Where does A think that B ...?' inwhich A and B are two agents. Otherwise, respond 'No'. \n\
        When the story only have one person, respond with 'No'. \n\
        Question: [A story involving several people.] Where will Jack look for the celery? \
        Higher-order belief: No \n\
        Question: [A story involving several people.] Where does Jack think that Chloe searches for the hat? \
        Higher-order belief: Yes \n\
        When the story only have one person, respond with 'No'. \n\
        Question: {question} \
        Higher-order belief: "
    response = gpt_call(prompt)
    if response == "Yes":
        return True
    else:
        return False

def modify_variables(assigned_model, vars, action="add"):
    modified_model = assigned_model[:]
    
    if action == "add":
        if vars == ["Action"]:
            vars.append("Goal")
        for var in vars:
            print('inside loop', var)
            assert var not in modified_model
            modified_model.append(var)
        modified_model = sorted(
            modified_model, key=all_variables.index
        )
    elif action == "remove":
        for var in vars:
            assert var in modified_model
            modified_model.remove(var)
    
    return modified_model


def clear_current_hypotheses(file_path, time):
    # Load the data
    df = pd.read_csv(file_path)
    df = df[df['Time'] != time]
    df.to_csv(file_path, index=False)
    return
