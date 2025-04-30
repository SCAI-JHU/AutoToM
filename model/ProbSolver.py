from BayesianInference import *
from ElementExtractor import *
from utils import *
from DataLoader import *
from Timeline import *
from model_adjustment import *
import numpy as np
import time
import utils
import probs
from scipy.stats import entropy
import NodeResultTracker
import TimestepInference
import ProblemParser
import Nested
import argparse

"""
    Creates a ProblemSolver class that will setup and answer the questions in the dataset 

    Args:
        story 
        

    Functions:
        init: initialize the BayesianInferenceModel 
        solve: sets up the general framework to solve a given question 
        main: run the ProblemSolver class to solve questions in a given dataset 
"""


def argmax(lst):
    return lst.index(max(lst))


def argmin(lst):
    return lst.index(min(lst))


class ProblemSolver:
    def __init__(
        self,
        story,
        question,
        choices,
        K,
        assigned_model=None,
        model_name="sobag",
        episode_name="",
        llm="gpt-4o",
        hypo_llm="gpt-4o",
        verbose=False,
        dataset_name=None,
        hypo_method=None,
        nested=False,
        tab="",
        full=True,
        video_id=None,
        answerfunc=None,
        back_inference=False,
        reduce_hypotheses=False,
        precomputed_states=None,
        precomputed_actions=None,
        prev_hyp=None,
        no_model_adjustment=False,
        recursion_depth=0,
        nested_timeline_before=None,
        nested_time_variables_before=None,
        init_belief=False,
    ):
        self.world_rules = (
            ""  # we do not use this value and keep it constant for all datasets
        )
        self.story = story
        self.question = question
        self.choices = choices
        self.K = K  # K is the number of hypotheses we want to propose
        self.assigned_model = deepcopy(assigned_model)
        self.llm = llm
        self.hypo_llm = hypo_llm
        self.model_name = model_name
        self.episode_name = episode_name
        self.verbose = verbose
        self.dataset_name = dataset_name
        self.hypo_method = hypo_method
        self.inf_agent_name = "NONE"
        self.nested_agents_list = []
        self.orig_choices = deepcopy(choices)
        self.orig_story = "NONE"
        self.nested = nested
        self.tab = tab
        self.full = full
        self.video_id = video_id
        self.answerfunc = answerfunc
        self.intermediate_node_results = []
        self.back_inference = back_inference
        self.reduce_hypotheses = reduce_hypotheses
        self.prev_hyp = prev_hyp
        self.estimation_dictionary = {}
        self.translate_id_recorder = {}
        self.start_time = time.time()
        self.middle_result_time = self.start_time
        self.no_model_adjustment = no_model_adjustment

        self.recursion_depth = recursion_depth

        self.nested_timeline_before = nested_timeline_before
        self.nested_time_variables_before = nested_time_variables_before

        self.start_cost = (
            probs.cost_of_estimating_likelihood
            + utils.cost_of_information_extracting
            + utils.cost_of_proposing_hypotheses
        )

        self.middle_result_cost = self.start_cost
        self.start_api = (
            probs.times_of_estimating
            + utils.times_of_information_extracting
            + utils.times_of_proposing_hypotheses
        )
        self.middle_api = self.start_api
        self.states = precomputed_states
        self.actions = precomputed_actions
        self.initial_state = "NONE"
        self.memory = False
        self.realistic = False
        self.ALL_VARIABLES = [
            "State",
            "Action",
            "Utterance",
            "Belief",
            "Goal",
            "Observation",
        ]  # all possbile variables for extraction
        self.init_belief = init_belief

        # import tracker and helper functions
        self.clear_current_nodes = NodeResultTracker.clear_current_nodes
        self.translate_and_add_node_results = (
            NodeResultTracker.translate_and_add_node_results
        )
        self.infer_last_timestamp = TimestepInference.infer_last_timestamp
        self.infer_belief_at_timestamp = TimestepInference.infer_belief_at_timestamp
        self.infer_goal_at_timestamp = TimestepInference.infer_goal_at_timestamp
        self.load_parsed_result_into_self = ProblemParser.load_parsed_result_into_self
        self.parse_story_and_question = ProblemParser.parse_story_and_question
        self.contains_utterance = utils.contains_utterance
        self.check_nested = utils.check_nested
        self.get_nested_states = Nested.get_nested_states
        self.save_nested_results = Nested.save_nested_results

    def information_extraction(self):
        """
        Extract states and actions, other mental variables that might appear in the text, as well as solved nested states.
        Sets up and processes time variables for the problem solver.

        Returns:
            tuple: Contains:
                - time_variables: List of variables at each timestep
                - variable_values_with_time: Timeline table of values
                - no_observation_hypothesis: hypothesis for no observation cases
                - all_timesteps: Total number of timesteps
        """
        context = ""

        if self.model_name == "automated":
            var_types = self.ALL_VARIABLES
        else:
            var_types = self.assigned_model

        # For belief of state questions for solving nested states, all episodes at same time step shares a timeline. We can load computed timeline (if computed before).
        if "nestedChunk" in self.episode_name:
            orig_episode_name = deepcopy(self.episode_name)
            self.episode_name = "_".join(
                self.episode_name.split("_")[:-1]
            )  # leave the "state hypotheses" at this time step

        timeline = TimeLine(
            self.story,
            self.question,
            self.choices,
            var_types,
            self.model_name,
            self.episode_name,
            self.inf_var_name,
            self.llm,
            self.dataset_name,
        )

        # Load timeline table
        variable_values_with_time = load_timeline_table(
            self.model_name, self.episode_name
        )
        if "MMToM" not in self.dataset_name:
            if variable_values_with_time == None:

                if self.model_name == "automated":
                    variable_values_with_time = load_timeline_table(
                        self.model_name, self.episode_name, reuse=True
                    )

                    if variable_values_with_time is None:
                        variable_values_with_time, init_belief = timeline.extract()
                        self.init_belief = init_belief
                    else:
                        variable_values_with_time, init_belief = (
                            timeline.supply_extraction(variable_values_with_time)
                        )
                        self.init_belief = init_belief

                else:
                    variable_values_with_time, init_belief = timeline.extract()
                    self.init_belief = init_belief
                if self.verbose:
                    print(variable_values_with_time)
            else:
                if (
                    (len(variable_values_with_time) == 1)
                    and f"{self.inf_agent_name}'s Action"
                    not in variable_values_with_time[0]
                ):
                    new_variable_values_with_time = deepcopy(variable_values_with_time)
                    for k in variable_values_with_time[0].keys():
                        if "'s Action" in k and self.inf_agent_name in k:
                            new_variable_values_with_time[0][
                                f"{self.inf_agent_name}'s Action"
                            ] = variable_values_with_time[0][k]
                    variable_values_with_time = new_variable_values_with_time

                if (
                    len(variable_values_with_time) == 1
                    and variable_values_with_time[0][f"{self.inf_agent_name}'s Action"]
                    == "NONE"
                ):
                    self.init_belief = True
                else:
                    self.init_belief = False

        if "nestedChunk" in self.episode_name:
            self.episode_name = orig_episode_name  # restore episode name for different belief hypotheses

        if self.nested:
            self.relevant_entities = find_relevant_entities(
                self.choices, self.nested_agents_list, self.llm
            )
            ground_truth_timeline = TimeLine(
                self.orig_story,
                self.question,
                self.orig_choices,
                self.assigned_model,
                self.model_name,
                f"{self.episode_name}_ground_truth",
                inf_var="Belief",
                llm=self.llm,
                dataset_name=self.dataset_name,
            )
            ground_truth_variable_values_with_time = load_timeline_table(
                self.model_name, f"{self.episode_name}_ground_truth"
            )
            if ground_truth_variable_values_with_time == None:
                ground_truth_variable_values_with_time, _ = (
                    ground_truth_timeline.extract()
                )
                if self.verbose:
                    print(ground_truth_variable_values_with_time)

        reuse = False
        if (
            self.nested == True
        ):  # We can try to reuse previously solved time variables for other models
            reuse = True

        if self.nested:  # final variables used for inference is stored in _depth2
            time_variables = load_time_variables(
                self.model_name, self.episode_name + "_depth2"
            )
        else:
            time_variables = load_time_variables(
                self.model_name, self.episode_name, reuse=reuse
            )

        variable_types = []
        for var_name in self.assigned_model:
            variable_types.append((self.inf_agent_name, var_name))
        variable_types.append(("", "All Actions"))

        if self.model_name == "automated":
            # For automated model, we only need to extract states and actions.
            # No need to propose hypotheses since we propose hypotheses at a later stage.
            variable_types = [
                ("", "State"),
                ("", "All Actions"),
                (self.inf_agent_name, "Action"),
            ]

        if time_variables == None:
            time_variables = get_variables_with_time(
                variable_values_with_time,
                variable_types,
                self.inf_agent_name,
                self.inf_var_name,
                context,
                self.choices,
                self.K,
                self.llm,
                self.hypo_llm,
                self.world_rules,
                self.verbose,
                self.hypo_method,
                self.dataset_name,
                self.full,
                self.initial_state,
                self.prev_hyp,
                self.states,
                self.actions,
                self.question,
            )

            if self.nested:
                # Logic: GT States -> Reconstructed but not solved States -> Solved belief of states
                reconstructed_but_not_solved_time_variables = deepcopy(time_variables)
                reconstructed_but_not_solved_variable_values_with_time = deepcopy(
                    variable_values_with_time
                )
                self.episode_name += "_depth" + str(len(self.nested_agents_list))
                ground_truth_time_variables = load_time_variables(
                    self.model_name, f"{self.episode_name}_gt"
                )
                if ground_truth_time_variables == None:
                    print(ground_truth_variable_values_with_time)
                    ground_truth_time_variables = get_variables_with_time(
                        ground_truth_variable_values_with_time,
                        variable_types,
                        self.inf_agent_name,
                        self.inf_var_name,
                        context,
                        self.orig_choices,
                        self.K,
                        self.llm,
                        self.hypo_llm,
                        self.world_rules,
                        self.verbose,
                        self.hypo_method,
                        self.dataset_name,
                        self.full,
                        self.initial_state,
                    )
                save_time_variables(
                    ground_truth_time_variables,
                    self.model_name,
                    f"{self.episode_name}_gt",
                )
                save_time_variables(
                    reconstructed_but_not_solved_time_variables,
                    self.model_name,
                    f"{self.episode_name}_reconstructed_but_not_solved",
                )
                solved_time_variables = load_time_variables(
                    self.model_name, self.episode_name
                )
                if solved_time_variables == None:
                    solved_time_variables = self.get_nested_states(
                        self,
                        reconstructed_but_not_solved_time_variables,
                        ground_truth_time_variables,
                        reconstructed_but_not_solved_variable_values_with_time,
                        ground_truth_variable_values_with_time,
                    )
                    save_time_variables(
                        solved_time_variables, self.model_name, self.episode_name
                    )
                while len(self.nested_agents_list) > 1:
                    self.nested_agents_list = self.nested_agents_list[1:]
                    if len(self.nested_agents_list) == 1:
                        time_variables = deepcopy(solved_time_variables)
                        break
                    self.first_agent_name = self.nested_agents_list[0]
                    self.story, vis = reconstruct_story_nested(
                        self.story, self.first_agent_name, self.llm, self.dataset_name
                    )
                    self.story = " ".join(self.story)

                    self.episode_name = self.episode_name[:-1] + str(
                        len(self.nested_agents_list)
                    )
                    save_reconstructed_story(
                        vis, self.model_name, self.episode_name, self.first_agent_name
                    )
                    reconstructed_but_not_solved_variable_values_with_time, _ = (
                        TimeLine(
                            self.story,
                            self.question,
                            self.choices,
                            self.assigned_model,
                            self.model_name,
                            self.episode_name,
                            self.inf_var_name,
                            self.llm,
                            self.dataset_name,
                        ).extract(inferred_agent=self.nested_agents_list[-1])
                    )
                    solved_time_variables = load_time_variables(
                        self.model_name, self.episode_name
                    )
                    if solved_time_variables is None:
                        reconstructed_but_not_solved_time_variables = (
                            get_variables_with_time(
                                reconstructed_but_not_solved_variable_values_with_time,
                                variable_types,
                                self.inf_agent_name,
                                self.inf_var_name,
                                context,
                                self.choices,
                                self.K,
                                self.llm,
                                self.hypo_llm,
                                self.world_rules,
                                self.verbose,
                                self.hypo_method,
                                self.dataset_name,
                                self.full,
                                self.initial_state,
                                self.prev_hyp,
                            )
                        )
                        solved_time_variables = self.get_nested_states(
                            self,
                            reconstructed_but_not_solved_time_variables,
                            ground_truth_time_variables,
                            reconstructed_but_not_solved_variable_values_with_time,
                            ground_truth_variable_values_with_time,
                        )
                        save_time_variables(
                            solved_time_variables, self.model_name, self.episode_name
                        )
                    ground_truth_variable_values_with_time = deepcopy(
                        reconstructed_but_not_solved_variable_values_with_time
                    )
                    ground_truth_time_variables = deepcopy(solved_time_variables)

            save_time_variables(time_variables, self.model_name, self.episode_name)

        # Update timing and cost metrics
        self.middle_result_time = time.time()
        self.middle_result_cost = (
            probs.cost_of_estimating_likelihood
            + utils.cost_of_information_extracting
            + utils.cost_of_proposing_hypotheses
        )
        self.middle_api = (
            probs.times_of_estimating
            + utils.times_of_information_extracting
            + utils.times_of_proposing_hypotheses
        )

        no_observation_hypothesis = "NONE"
        for variables in time_variables:
            for key, val in variables.items():
                if "Observation" in key and val.is_observed is False:
                    no_observation_hypothesis = val.possible_values[-1]
                    break

        all_timesteps = len(time_variables)

        return (
            time_variables,
            variable_values_with_time,
            no_observation_hypothesis,
            all_timesteps,
        )

    def solve_with_automated_model(
        self,
        time_variables,
        all_timesteps,
        no_observation_hypothesis,
        variable_values_with_time,
        all_probs,
    ):

        preproposed_ob_hypos = []
        last_state = "None"

        precomputed_states = []
        for i in range(len(time_variables)):
            precomputed_states.append(time_variables[i]["State"].possible_values[0])

        inf_agent_action = "NONE"
        belief_name = f"{self.inf_agent_name}'s Belief"
        inf_name = f"{self.inf_agent_name}'s {self.inf_var_name}"

        # print("time_variables", time_variables)
        # print("variable_values_with_time", variable_values_with_time)
        contain_utterance = self.contains_utterance(
            self, time_variables, variable_values_with_time
        )
        proposed_model = initial_model_proposal(
            self.story + self.question,
            self.inf_var_name,
            self.nested,
            contain_utterance,
        )
        assigned_models = {}
        saved_model_variables = {}  # with regard to timestep

        for start_timestep in range(all_timesteps - 1, -1, -1):
            print(f"Starting from timestep {start_timestep}")
            if start_timestep > 0 and belief_name in time_variables[start_timestep - 1]:
                previous_belief = deepcopy(
                    time_variables[start_timestep - 1][belief_name]
                )
                previous_belief.name = "Previous Belief"
            else:
                previous_belief = Variable(
                    "Previous Belief", True, False, ["NONE"], np.ones(1)
                )

            output_folder = "../results/node_results"
            file_path = f"{output_folder}/automated_{self.episode_name}_back{int(self.back_inference)}_reduce{int(self.reduce_hypotheses)}.csv"

            assigned_models[start_timestep] = proposed_model
            results, terminate, assigned_models, saved_model_variables = (
                model_discovery(
                    start_timestep,
                    all_timesteps,
                    self.verbose,
                    time_variables,
                    previous_belief,
                    self.inf_agent_name,
                    self.inf_var_name,
                    self.estimation_dictionary,
                    self.infer_last_timestamp,
                    no_observation_hypothesis,
                    variable_values_with_time,
                    all_probs,
                    self.answerfunc,
                    argmax,
                    argmin,
                    save_belief_probs,
                    self.model_name,
                    self.episode_name,
                    self.infer_belief_at_timestamp,
                    belief_name,
                    get_variables_at_time,
                    mmtom_get_variables_at_time,
                    self.choices,
                    self.K,
                    self.llm,
                    self.hypo_llm,
                    self.hypo_method,
                    self.full,
                    preproposed_ob_hypos,
                    last_state,
                    inf_agent_action,
                    assigned_models,
                    file_path,
                    self.clear_current_nodes,
                    self.dataset_name,
                    self.states,
                    self.actions,
                    self.question,
                    precomputed_states,
                    saved_model_variables,
                    self.no_model_adjustment,
                    self,
                )
            )
            if terminate:
                model_record = {
                    "Initial model propose": proposed_model,
                    "Assigned models": assigned_models,
                }
                print(model_record)
                return results, model_record

        model_record = {
            "Initial model propose": proposed_model,
            "Assigned models": assigned_models,
        }
        print(model_record)
        return results, model_record

    def solve(self):
        if len(self.choices) == 1:
            return [1.0], {}
        ### Parsing ###
        if self.model_name == "automated":
            if determine_realistic_questions(self.question) is True:
                self.realistic = True
                self.nested = False
            if self.nested == None:
                self.nested = determine_higher_order_belief(self.story + self.question)
                print("nested", self.nested)
                if self.check_nested(self) == False:
                    return None, {}
            if determine_memory_questions(self.question) is True:
                self.memory = True

        parsed_result = load_parsed_result(self.model_name, self.episode_name)
        if parsed_result is None:
            parsed_result = self.parse_story_and_question(self)
        else:
            self.load_parsed_result_into_self(self, parsed_result)

        if self.memory is True:
            return get_answer_memory_questions(
                self.story, self.question, self.choices, self.llm
            )

        ### Extract states, actions, and other assigned variables ###
        (
            time_variables,
            variable_values_with_time,
            no_observation_hypothesis,
            all_timesteps,
        ) = self.information_extraction()

        if self.realistic:
            return get_answer_from_state(
                time_variables[-1]["State"].possible_values[0], self.choices, self.llm
            )

        previous_belief = Variable("Previous Belief", True, False, ["None"], np.ones(1))
        all_probs = []

        self.estimation_dictionary = load_estimation_dict(self.dataset_name)
        results = None
        action_likelihood_goal = {}

        if self.model_name == "automated":
            return self.solve_with_automated_model(
                time_variables,
                all_timesteps,
                no_observation_hypothesis,
                variable_values_with_time,
                all_probs,
            )

        else:
            if not args.back_inference:
                start_timestep = 0
                print(f"Starting from timestep {start_timestep}")

                belief_name = f"{self.inf_agent_name}'s Belief"
                previous_belief = Variable(
                    "Previous Belief", True, False, ["NONE"], np.ones(1)
                )

                for i in range(start_timestep, all_timesteps):
                    if self.verbose:
                        print(f"------------- time stamp {i} -------------")
                    time_variables[i]["Previous Belief"] = previous_belief
                    variables = time_variables[i]
                    now_variables = []
                    inf_name = f"{self.inf_agent_name}'s {self.inf_var_name}"

                    if i == all_timesteps - 1:
                        results, self.estimation_dictionary, all_probs = (
                            self.infer_last_timestamp(
                                self,
                                time_variables=time_variables,
                                i=i,
                                inf_name=inf_name,
                                inf_var_name=self.inf_var_name,
                                now_variables=now_variables,
                                no_observation_hypothesis=no_observation_hypothesis,
                                variable_values_with_time=variable_values_with_time,
                                all_probs=all_probs,
                                all_prob_estimations=self.estimation_dictionary,
                                action_likelihood_goal=action_likelihood_goal
                            )
                        )
                        results = list(results)
                        save_belief_probs(
                            all_probs, self.model_name, self.episode_name
                        )
                        return results, {}

                    else:
                        if self.inf_var_name == "Goal":
                            action_likelihood_goal[i], self.estimation_dictionary, all_probs = (
                                self.infer_goal_at_timestamp(
                                    self,
                                    time_variables=time_variables,
                                    i=i,
                                    previous_belief=previous_belief,
                                    belief_name=belief_name,
                                    variable_values_with_time=variable_values_with_time,
                                    all_probs=all_probs,
                                    no_observation_hypothesis=no_observation_hypothesis,
                                    all_prob_estimations=self.estimation_dictionary,
                                    goal_name=f"{self.inf_agent_name}'s Goal"
                                )
                            )
                        previous_belief, self.estimation_dictionary, all_probs = (
                            self.infer_belief_at_timestamp(
                                self,
                                time_variables=time_variables,
                                i=i,
                                previous_belief=previous_belief,
                                belief_name=belief_name,
                                variable_values_with_time=variable_values_with_time,
                                all_probs=all_probs,
                                no_observation_hypothesis=no_observation_hypothesis,
                                all_prob_estimations=self.estimation_dictionary,
                            )
                        )
            else:
                # AutoToM w/ given specified model input
                for start_timestep in range(all_timesteps - 1, -1, -1):
                    print(f"Starting from timestep {start_timestep}")

                    belief_name = f"{self.inf_agent_name}'s Belief"

                    # If we have actual hypotheses for previous belief, include them in the model, but with no prior
                    if (
                        start_timestep > 0
                        and belief_name in time_variables[start_timestep - 1]
                    ):
                        previous_belief = deepcopy(
                            time_variables[start_timestep - 1][belief_name]
                        )
                    else:
                        previous_belief = Variable(
                            "Previous Belief", True, False, ["NONE"], np.ones(1)
                        )

                    for i in range(start_timestep, all_timesteps):
                        if self.verbose:
                            print(f"------------- time stamp {i} -------------")
                        time_variables[i]["Previous Belief"] = previous_belief
                        variables = time_variables[i]
                        now_variables = []
                        inf_name = f"{self.inf_agent_name}'s {self.inf_var_name}"

                        if i == all_timesteps - 1:
                            results, self.estimation_dictionary, all_probs = (
                                self.infer_last_timestamp(
                                    self,
                                    time_variables=time_variables,
                                    i=i,
                                    inf_name=inf_name,
                                    inf_var_name=self.inf_var_name,
                                    now_variables=now_variables,
                                    no_observation_hypothesis=no_observation_hypothesis,
                                    variable_values_with_time=variable_values_with_time,
                                    all_probs=all_probs,
                                    all_prob_estimations=self.estimation_dictionary,
                                    action_likelihood_goal=action_likelihood_goal
                                )
                            )

                            # determine if we can stop inference
                            results = list(results)
                            terminate = False

                            utility_terminate_threshold = -0.673
                            utility = -entropy(results)
                            if len(results) == 2:
                                if utility > utility_terminate_threshold:
                                    terminate = True
                            else:
                                if (
                                    self.answerfunc == argmax
                                    and utility > utility_terminate_threshold
                                ):
                                    terminate = True
                                elif self.answerfunc == argmin and min(results) < 0.2:
                                    terminate = True

                            if terminate is True:
                                save_belief_probs(
                                    all_probs, self.model_name, self.episode_name
                                )
                                return results, {}

                        else:
                            if self.inf_var_name == "Goal":
                                action_likelihood_goal[i], self.estimation_dictionary, all_probs = (
                                    self.infer_goal_at_timestamp(
                                        self,
                                        time_variables=time_variables,
                                        i=i,
                                        previous_belief=previous_belief,
                                        belief_name=belief_name,
                                        variable_values_with_time=variable_values_with_time,
                                        all_probs=all_probs,
                                        no_observation_hypothesis=no_observation_hypothesis,
                                        all_prob_estimations=self.estimation_dictionary,
                                        goal_name=f"{self.inf_agent_name}'s Goal"
                                    )
                                )
                            previous_belief, self.estimation_dictionary, all_probs = (
                                self.infer_belief_at_timestamp(
                                    self,
                                    time_variables=time_variables,
                                    i=i,
                                    previous_belief=previous_belief,
                                    belief_name=belief_name,
                                    variable_values_with_time=variable_values_with_time,
                                    all_probs=all_probs,
                                    no_observation_hypothesis=no_observation_hypothesis,
                                    all_prob_estimations=self.estimation_dictionary,
                                )
                            )


        return results, {}


def main(args):
    """
    Runs the ProblemSolver class to solve questions in a given dataset

    Args:
        dataset_name (str): name of the datasets (e.g. MMToM-QA, ToMi)
        K (int): number of hypotheses you want to propose for each variable
        llm (str): name of the LLM you want to use for inference + hypothesis proposals
        assigned_model (list): list of the variables that you want to use in the Bayesian Inference model

    Returns:
        Prints the number of questions correct and the correctness of each question
    """
    dataset_name = args.dataset_name
    data = load_full_dataset(args.dataset_name)

    # data = load_dataset(dataset_name)
    cnt = 0
    correct = []
    K = args.K
    assigned_model = args.assigned_model
    automated = args.automated
    no_model_adjustment = args.no_model_adjustment  # ablation study
    nested = args.nested
    # nested = False
    if automated:
        model_name = "automated"
    else:
        model_name = get_model_name(assigned_model)
    print("model_name:", model_name)

    llm = args.llm_model  # LLM
    hypo_generation_methods_list = [
        "None",
        "domain_specific",
        "guided",
        "infer_world_rules",
    ]
    hypo_method = "guided"
    print(hypo_method)

    order = 0
    if "HiToM" in dataset_name:
        order = int(dataset_name.split("order")[1])

    correct_answer_probs = []
    model_records = {}

    back_inference = args.back_inference
    print(f"Back inference is {back_inference}")
    reduce_hypotheses = args.reduce_hypotheses
    print(f"Reduce observation hypotheses is {reduce_hypotheses}")
    costs = []
    apis = []
    for i, d in enumerate(data):
        if i != 244:
            continue
        print(f"Question {i}")
        states, actions, video_id = None, None, None
        if "MuMa" in dataset_name:
            story, question, choices, correct_answer, video_id = d
        elif "MMToM" in dataset_name:
            story, question, choices, correct_answer, states, actions = d
        else:
            story, question, choices, correct_answer = d
        # video_input = get_base64_images(video_id, frames=16)

        if "HiToM" in dataset_name:
            # Deal with a lot of unnecessary choices
            orig_choices = deepcopy(choices)
            choices = []
            for j, c in enumerate(orig_choices):
                if c in story:
                    choices.append(c)

        answerfunc = argmax
        if "LEAST likely" in question:
            answerfunc = argmin

        solver = ProblemSolver(
            story=story,
            question=question,
            choices=choices,
            K=K,
            assigned_model=assigned_model,
            model_name=model_name,
            episode_name=f"{dataset_name}_{i}",
            llm=llm,
            verbose=True,
            dataset_name=dataset_name,
            hypo_method=hypo_method,
            nested=nested,
            video_id=video_id,
            answerfunc=answerfunc,
            back_inference=back_inference,
            reduce_hypotheses=reduce_hypotheses,
            precomputed_states=states,
            precomputed_actions=actions,
            prev_hyp=None,
            no_model_adjustment=no_model_adjustment,
            recursion_depth=order,
        )

        final_probs, model_record = solver.solve()
        model_records[f"Question {i}"] = model_record

        end_time = time.time()
        end_cost = (
            probs.cost_of_estimating_likelihood
            + utils.cost_of_information_extracting
            + utils.cost_of_proposing_hypotheses
        )
        end_api = (
            probs.times_of_estimating
            + utils.times_of_information_extracting
            + utils.times_of_proposing_hypotheses
        )

        save_estimation_dict(dataset_name, solver.estimation_dictionary)

        if final_probs == None:
            print("The assigned model cannot answer the question.")
            correct.append(0)
            enh_print(f"Incorrect, now {cnt} / {i + 1}, {correct}", "red")
            continue

        answer_idx = answerfunc(final_probs)
        if "HiToM" in dataset_name:
            true_answer_word = orig_choices[correct_answer]
            id = -1
            # print(correct_answer)
            for j, c in enumerate(choices):
                # print(c, true_answer_word)
                if c == true_answer_word:
                    id = j
        else:
            id = letter_to_number_mapping(correct_answer)
        correct_answer_prob = final_probs[id]

        correct_answer_probs.append(correct_answer_prob)
        enh_print(f"Likelihood of the correct answer: {correct_answer_prob:.2f}")

        metrics = {
            "Likelihood of correct answer": correct_answer_prob,
            "Total time": end_time - solver.start_time,
            "Middle time": solver.middle_result_time - solver.start_time,
            "Inference time": end_time - solver.middle_result_time,
            "Total cost": end_cost - solver.start_cost,
            "Middle cost": solver.middle_result_cost - solver.start_cost,
            "Total API": end_api - solver.start_api,
            "Middle API": solver.middle_api - solver.start_api,
            "probs": final_probs,
            "Correctness": answer_idx == id,
        }
        save_metrics(
            metrics,
            solver.model_name,
            solver.episode_name,
            solver.back_inference,
            solver.reduce_hypotheses,
        )
        costs.append(end_cost)
        apis.append(end_api)
        enh_print(f"Costs: {costs}", "red")
        enh_print(f"Apis: {apis}", "red")
        if answer_idx == id:
            cnt += 1
            correct.append(1)
            enh_print(f"Correct, now {cnt} / {i + 1}, {correct}")
        else:
            correct.append(0)
            enh_print(f"Incorrect, now {cnt} / {i + 1}, {correct}", "red")

    if automated:
        model_counts = {}
        for record in model_records.values():
            print("Record:", record)
            assigned_models = record["Assigned models"]
            for model_list in assigned_models.values():
                model_tuple = tuple(model_list)
                model_counts[model_tuple] = model_counts.get(model_tuple, 0) + 1

        print("Model records:", model_records)
        print("Model counts:", model_counts)

    # print("Correct answer probabilities:", correct_answer_probs)
    print("Costs:", costs)
    print("Apis:", apis)
    print("Number of correctness:", correct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=[
            "MMToM-QA",
            "ToMi-1st",
            "ToMi-2nd",
            "ToMi-memory",
            "ToMi-reality",
            "BigToM_fatb", "BigToM_fafb",
            "BigToM_fbtb",
            "BigToM_fbfb", "BigToM_bbfb",
            "BigToM_bbfb",
            "MuMaToM_social_goal",
            "MuMaToM_belief",
            "MuMaToM_belief_of_goal",
            "HiToM_len1_tell0_order0",
            "HiToM_len1_tell0_order1",
            "HiToM_len1_tell0_order2",
            "HiToM_len1_tell0_order3",
            "HiToM_len1_tell0_order4",
            "HiToM_len1_tell1_order0",
            "HiToM_len1_tell1_order1",
            "HiToM_len1_tell1_order2",
            "HiToM_len1_tell1_order3",
            "HiToM_len1_tell1_order4",
        ],
    )

    parser.add_argument(
        "--llm_model",
        choices=[
            "gpt-4o",
        ],
        default="gpt-4o"
    )
    parser.add_argument("--automated", action="store_true", help="Run automated model.")
    parser.add_argument(
        "--no_back_inference",
        action='store_true',
        default=False,
        help="Enable running AutoToM with backwards inference.",
    )
    parser.add_argument(
        "--reduce_hypotheses", type=bool, default=True, 
        help="Flag for running AutoToM with reduced hypotheses.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Flag for verbose.",
    )
    parser.add_argument(
        "--no_model_adjustment",
        action="store_true",
        help="Ablation studies"
    )
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--max_num", type=int, default=3)
    parser.add_argument(
        "--assigned_model",
        type=str,
        default='["State", "Observation", "Belief", "Action", "Goal"]',
        help="When automated is false, you can assign a manually defined model here."
    )
    parser.add_argument("--nested", default=None, help="If None, the model will figure out the order itself.")
    args = parser.parse_args()
    args.assigned_model = eval(args.assigned_model)
    args.back_inference = not args.no_back_inference
    print(args)
    main(args)
    # example of command line run with AutoToM, back inference, reduced hypotheses:
    # python ProbSolver.py --automated --dataset_name "MMToM-QA"

    # example of command line run with AutoToM with model specs:
    # python ProbSolver.py --dataset_name ToMi-1st --assigned_model "['State', 'Observation', 'Belief']"
