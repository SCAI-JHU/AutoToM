
def clear_current_nodes(self, start_timestep):

    self.intermediate_node_results = [
        node
        for node in self.intermediate_node_results
        if node["Time"] != start_timestep
    ]

def translate_and_add_node_results(self, i, all_node_results):
    id_recorder = self.translate_id_recorder
    for res in all_node_results:
        son, parents, var_dict, logits = res
        node_name = f"{son}_{i}"
        node_value = var_dict[son]

        if node_name not in id_recorder:
            id_recorder[node_name] = {"DIFFERENT_HYPOS": 0}
        if node_value not in id_recorder[node_name]:
            id_recorder[node_name][node_value] = (
                id_recorder[node_name]["DIFFERENT_HYPOS"] + 1
            )
            id_recorder[node_name]["DIFFERENT_HYPOS"] += 1
        node_name += f"_{id_recorder[node_name][node_value]}"

        parents_node_name = []
        parents_node_value = []
        for p in parents:
            newname = f"Belief_{i - 1}" if p == "Previous Belief" else f"{p}_{i}"

            if p == "Previous Belief" and i == 0:
                continue

            if newname not in id_recorder:
                id_recorder[newname] = {"DIFFERENT_HYPOS": 0}
            if var_dict[p] not in id_recorder[newname]:
                id_recorder[newname][var_dict[p]] = (
                    id_recorder[newname]["DIFFERENT_HYPOS"] + 1
                )
                id_recorder[newname]["DIFFERENT_HYPOS"] += 1

            newname += f"_{id_recorder[newname][var_dict[p]]}"

            parents_node_name.append(newname)
            parents_node_value.append(var_dict[p])

        node_dict = {
            "Time": i,
            "Node": node_name,
            "Parent node": parents_node_name,
            "Likelihood": logits,
            "Node value": node_value,
            "Parent node value": parents_node_value,
        }

        if node_dict not in self.intermediate_node_results:
            self.intermediate_node_results.append(node_dict)
