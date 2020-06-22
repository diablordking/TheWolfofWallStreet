from functions import *


def ant_miner(dataset, no_of_ants, min_case_per_rule, max_uncovered_cases, no_rules_converg, fold):
    log_file = "log_main.txt"
    i_log_file = "log_colony-loop.txt"
    f = open(log_file, "a+")
    f.write('\n\n\n==================== EXTERNAL LOOP ==========================================================================')
    f.write('\n> Stopping condition: number of uncovered cases')
    f.write('\n> Initialise pheromones > create t rule > choose best rule > add to the list > remove rule covered cases')
    f.write('\n============================================================================================================')
    f.write('\n FOLD OF CROSS VALIDATION: ' + repr(fold))
    f.close()

    training_dataset = copy.deepcopy(dataset)
    discovered_rule_list = []

    dataset_stagnation_test = 0
    no_of_remaining_cases = len(training_dataset.data)

    idx_e = 0
    while no_of_remaining_cases > max_uncovered_cases:
        idx_e += 1
        f = open(log_file, "a+")
        f.write('\n\n>>>>>>>>>>>>>>> ITERATION ' + repr(idx_e))
        f.close()

        previous_rule = Rule(training_dataset)
        ant_index = 0
        converg_test_index = 1

        list_of_current_rules = []
        list_of_current_rules_quality = []

        list_of_terms = get_terms(training_dataset.attr_values)
        list_of_terms = set_pheromone_init(list_of_terms)
        list_of_terms = set_heuristic_values(list_of_terms, training_dataset)

        # DATASET STAGNATION >> REVIEW NECESSITY
        last_no_of_remaining_cases = no_of_remaining_cases
        if no_of_remaining_cases == last_no_of_remaining_cases:
            dataset_stagnation_test += 1
        if dataset_stagnation_test == no_rules_converg:
            f = open(log_file, "a+")
            f.write('\n\n==================== END EXTERNAL LOOP')
            f.write('\n!! Alternative Condition: dataset stagnation')
            f.write('\n   - no_of_remaining_cases = ' + repr(no_of_remaining_cases))
            f.write('\n   - dataset_stagnation_test = ' + repr(dataset_stagnation_test))
            f.close()
            break

        f = open(log_file, "a+")
        f.write('\n> Number of terms: ' + repr(len(list_of_terms)))
        f.write('\n\n=> Internal Loop procedure: colony-loop_log-results.txt file <=\n')
        f.close()

        f = open(i_log_file, "a+")
        f.write('\n\n\n=================== INTERNAL LOOP ==========================================================================')
        f.write('\n> Stopping condition: no_of_ants || no_rules_converg')
        f.write('\n> Incremental rule construction > rule pruning > pheromone updating > convergence test')
        f.write('\n============================================================================================================')
        f.write('\n EXTERNAL LOOP ITERATION ' + repr(idx_e))
        f.close()

        idx_i = 0
        while True:
            idx_i += 1
            f = open(i_log_file, "a+")
            f.write('\n\n>>>>>>>>>>>>>>> ITERATION ' + repr(idx_i))
            f.close()

            if ant_index >= no_of_ants or converg_test_index >= no_rules_converg:
                f = open(i_log_file, "a+")
                f.write('\n\n==================== END Internal Loop')
                f.write('\n  - no_of_ants = ' + repr(no_of_ants))
                f.write('\n  - ant_index = ' + repr(ant_index))
                f.write('\n  - no_rules_converg = ' + repr(no_rules_converg))
                f.write('\n  - converg_test_index = ' + repr(converg_test_index))
                f.write('\n  - Number of iterations: ' + repr(idx_i))
                f.close()
                break

            # RULE CONSTRUCTION
            f = open(i_log_file, "a+")
            f.write('\n\n=> Rule Construction Function: rule-construction-fnc_log-results.txt file <=')
            f.close()
            current_rule = rule_construction(list_of_terms, min_case_per_rule, training_dataset, idx_e, idx_i)

            # Case Rule-Constructed is NONE >> check necessity !!!
            if current_rule is None:
                f = open(i_log_file, "a+")
                f.write('\n!!! Empty Rule Constructed !!!')
                f.close()
                # print('Empty rule')
                ant_index += 1
                converg_test_index += 1
                continue

            # RULE PRUNING
            f = open(i_log_file, "a+")
            f.write('\n\n=> Rule Pruning Function: rule-pruning-fnc_log-results.txt file <=')
            f.close()
            current_rule_pruned = rule_pruning(current_rule, min_case_per_rule, training_dataset, idx_e, idx_i)

            f = open(i_log_file, "a+")
            f.write('\n\n> Rule Constructed:')
            f.close()
            current_rule.print_txt(i_log_file, 'Class')
            f = open(i_log_file, "a+")
            f.write('\n-Quality: ' + repr(current_rule.quality))
            f.write('\n\n> Rule Pruned:')
            f.close()
            current_rule_pruned.print_txt(i_log_file, 'Class')
            f = open(i_log_file, "a+")
            f.write('\n-Quality: ' + repr(current_rule_pruned.quality))
            f.close()

            # just for log register
            if len(list_of_current_rules) >= 1:
                last_list_rule = list_of_current_rules[-1]
                f = open(i_log_file, "a+")
                f.write('\n\n> Last constructed-pruned rule from current_rule_list:')
                f.close()
                last_list_rule.print_txt(i_log_file, 'Class')
                f = open(i_log_file, "a+")
                f.write('\n-Quality: ' + repr(last_list_rule.quality))
                f.close()

            # converg_test_index = check_convergence(current_rule_pruned, list_of_current_rules, converg_test_index)
            if current_rule_pruned.equals(previous_rule):
                converg_test_index += 1
            else:
                list_of_current_rules.append(current_rule_pruned)
                list_of_current_rules_quality.append(current_rule_pruned.quality)
                converg_test_index = 1
                previous_rule = copy.deepcopy(current_rule_pruned)
                f = open(i_log_file, "a+")
                f.write('\n\n!!! Pruned Constructed Rule did not converged')
                f.write('\n!!! Pruned Rule added to current_rule_list')
                f.close()

            list_of_terms = pheromone_updating(list_of_terms, current_rule_pruned)
            ant_index += 1
        # END OF COLONY LOOP

        # !!! CHECK NECESSITY
        if not list_of_current_rules_quality:
            f = open(i_log_file, "a+")
            f.write('\n\n!!! WARNING: Internal Loop added no rule quality to list_of_current_rules_quality > continue')
            f.close()
            continue  # GOES BACK TO LOOP WHILE UNCOVERED CASES

        best_rule_idx = list_of_current_rules_quality.index(max(list_of_current_rules_quality))
        best_rule = copy.deepcopy(list_of_current_rules[best_rule_idx])
        discovered_rule_list.append(best_rule)

        covered_cases = np.array(best_rule.covered_cases)
        training_dataset.data = np.delete(training_dataset.data, covered_cases, axis=0)
        training_dataset.data_updating()
        no_of_remaining_cases = len(training_dataset.data)

        # just for log register
        f = open(log_file, "a+")
        f.write('\n>> Internal Loop Results:')
        f.write('\n>Number of generated rules on internal loop: ' + repr(len(list_of_current_rules)))
        f.write('\n>Generated rules-quality list: ' + repr(list_of_current_rules_quality))
        f.write('\n>Best rule index: ' + repr(best_rule_idx))
        f.write('\n>Best rule:')
        f.close()
        best_rule.print_txt(log_file, 'Class')
        f = open(log_file, "a+")
        f.write('\n- number of covered cases: ' + repr(len(covered_cases)))
        f.write('\n\n>> External Loop Information:')
        f.write('\n>Number of rules on discovered_rule_list: ' + repr(len(discovered_rule_list)))
        f.write('\n>Last_no_of_remaining_cases: ' + repr(last_no_of_remaining_cases))
        f.write('\n>No_of_remaining_cases: ' + repr(no_of_remaining_cases))
        f.close()
    # END OF WHILE (AVAILABLE_CASES > MAX_UNCOVERED_CASES)

    # just for log register
    f = open(log_file, "a+")
    f.write('\n\n==================== END EXTERNAL LOOP: end of Ant-Miner Algorithm')
    f.write('\n> Stopping Condition: number of remaining cases')
    f.write('\n   - no_of_remaining_cases = ' + repr(no_of_remaining_cases))
    f.write('\n   - max_uncovered_cases = ' + repr(max_uncovered_cases))
    f.close()

    # generating rule for remaining cases
    rule_for_remaining_cases = get_remaining_cases_rule(training_dataset)
    discovered_rule_list.append(rule_for_remaining_cases)

    return discovered_rule_list, training_dataset, no_of_remaining_cases

