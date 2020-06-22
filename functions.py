import copy
import collections
import numpy as np
from terms import Terms
from rule import Rule


def get_terms(dict_attr_values):

    list_of_terms = []
    idx = 0
    for a in dict_attr_values:
        for v in dict_attr_values[a]:
            term_obj = Terms()
            term_obj.attribute = a
            term_obj.value = v
            term_obj.term_idx = idx
            list_of_terms.append(term_obj)
            idx += 1

    return list_of_terms


def set_pheromone_init(list_of_terms):

    for term in list_of_terms:
        term.pheromone = 1/len(list_of_terms)

    return list_of_terms


def set_heuristic_values(list_of_terms, dataset):

    terms_logK_entropy = []
    for term in list_of_terms:
        term.set_entropy(dataset)
        terms_logK_entropy.append(term.logK_entropy)

    for term in list_of_terms:
        term.set_heuristic(len(dataset.class_values), sum(terms_logK_entropy))

    return list_of_terms


def set_probability_values(list_of_terms):

    denominator = 0

    for term in list_of_terms:
        den = term.heuristic * term.pheromone
        denominator = denominator + den

    for term in list_of_terms:
        term.set_probability(denominator)

    return list_of_terms


def sort_term(list_of_terms, c_log_file):

    f = open(c_log_file, "a+")
    f.write('\n\n> TERM SORT:')
    f.close()

    term_chosen = None
    index = None

    terms_prob_sum = 0
    for term in list_of_terms:
        terms_prob_sum = terms_prob_sum + term.probability

    if terms_prob_sum == 0:
        return term_chosen, index

    number_sort = np.random.rand()
    f = open(c_log_file, "a+")
    f.write('\n- Sorted random number: ' + repr(number_sort))
    f.close()

    f = open(c_log_file, "a+")
    f.write('\n\nSorting...:')
    f.close()
    probabilities_sort = 0
    for term_idx, term in enumerate(list_of_terms):

        prob_norm = term.probability/terms_prob_sum
        probabilities_sort = probabilities_sort + prob_norm

        f = open(c_log_file, "a+")
        f.write('\n-Term ' + repr(term_idx) + ': Term_prob=' + repr(prob_norm) + ' Accum_Prob=' + repr(probabilities_sort))
        f.close()

        if number_sort <= probabilities_sort:
            term_chosen = term
            index = list_of_terms.index(term)
            break

    f = open(c_log_file, "a+")
    f.write('\n\n-> Term Sorted: Term ' + repr(index))
    f.close()

    return term_chosen, index


def list_terms_updating(list_of_terms, attribute):

    new_list = []
    for term in list_of_terms:
        if term.attribute != attribute:
            new_list.append(term)
            # list_of_terms.pop(term)

    return new_list


def rule_construction(list_of_terms, min_case_per_rule, dataset, idx_e, idx_i):
    c_log_file = "log_rule-construction-fnc.txt"
    f = open(c_log_file, "a+")
    f.write('\n\n\n=============== RULE CONSTRUCTION LOOP ======================================================================')
    f.write('\n> Stopping condition: rule cover less than minimum cases or there is no more attributes to be added')
    f.write('\n> Sequential construction: Sort a term to be added to rule > check number of covered cases ')
    f.write('\n> IF Stopping condition: set rule consequent')
    f.write('\n=============================================================================================================')
    f.write('\n EXTERNAL LOOP ITERATION ' + repr(idx_e))
    f.write('\n INTERNAL LOOP ITERATION ' + repr(idx_i))
    f.close()
    idx = 0

    new_rule = Rule(dataset)
    current_list_of_terms = copy.deepcopy(list_of_terms)

    # Antecedent construction
    while True:
        idx += 1
        f = open(c_log_file, "a+")
        f.write('\n\n>>>>>>>>>>>>>>> ITERATION ' + repr(idx))
        f.close()
        f = open(c_log_file, "a+")
        f.write('\n\n==> SEQUENTIAL CONSTRUCTION:')
        f.write('\n> List_of_terms size: ' + repr(len(current_list_of_terms)))
        f.close()

        previous_rule = copy.deepcopy(new_rule)

        if not current_list_of_terms:
            f = open(c_log_file, "a+")
            f.write('\n\n=============== END CONSTRUCTION')
            f.write('\n> Condition: empty terms list')
            f.write('\n   - current_list_of_terms size = ' + repr(len(current_list_of_terms)))
            f.write('\n   - iteration number = ' + repr(idx))
            f.close()
            break

        # Sorting term
        current_list_of_terms = set_probability_values(current_list_of_terms)
        term_2b_added, term_2b_added_index = sort_term(current_list_of_terms, c_log_file)

        if term_2b_added is None:   # !!! CHECK NECESSITY
            f = open(c_log_file, "a+")
            f.write('\n\n>>>>> END Construction')
            f.write('\n!! Alternative Condition: empty term_2b_added')
            f.close()
            break
        f = open(c_log_file, "a+")
        f.write('\n\n> TERM TO BE ADDED: Attribute=' + repr(term_2b_added.attribute) + ' Value=' + repr(term_2b_added.value))
        f.close()

        # Adding term and updating rule-obj
        new_rule.antecedent[term_2b_added.attribute] = term_2b_added.value
        new_rule.added_terms.append(term_2b_added)
        new_rule.set_covered_cases(dataset)

        f = open(c_log_file, "a+")
        f.write('\n\n==> CONSTRUCTION ITERATION ' + repr(idx) + ' RESULTS:')
        f.write('\n- Constructed Rule:')
        f.close()
        new_rule.print_txt(c_log_file, 'Class')
        f = open(c_log_file, "a+")
        f.write('\n- Previous Rule:')
        f.close()
        previous_rule.print_txt(c_log_file, 'Class')

        if new_rule.no_covered_cases < min_case_per_rule:
            f = open(c_log_file, "a+")
            f.write('\n\n=============== END CONSTRUCTION')
            f.write('\n> Condition: constructed_rule.no_covered_cases < min_case_per_rule')
            f.write('\n\n> Last constructed rule: (condition = true)')
            f.close()
            new_rule.print_txt(c_log_file, 'Class')
            f = open(c_log_file, "a+")
            f.write('\n-no_covered_cases: ' + repr(new_rule.no_covered_cases))
            f.close()

            new_rule = copy.deepcopy(previous_rule)

            f = open(c_log_file, "a+")
            f.write('\n\n> Previous constructed rule:')
            f.close()
            new_rule.print_txt(c_log_file, 'Class')
            f = open(c_log_file, "a+")
            f.write('\n-no_covered_cases: ' + repr(new_rule.no_covered_cases))
            f.close()
            break

        current_list_of_terms = list_terms_updating(current_list_of_terms, term_2b_added.attribute)

    if not new_rule.antecedent:
        f = open(c_log_file, "a+")
        f.write('\n\n>>>>> WARNING')
        f.write('\n!! No rule antecedent constructed')
        f.write('\n  - Number of iterations: ' + repr(idx))
        f.close()
        return None

    # Consequent selection
    new_rule.set_consequent(dataset)
    new_rule.set_quality(dataset, idx_e, idx_i, p=False)

    f = open(c_log_file, "a+")
    f.write('\n\n>>> FINAL RULE ')
    f.close()
    new_rule.print_txt(c_log_file, 'Class')
    f = open(c_log_file, "a+")
    f.write('\n-no_covered_cases: ' + repr(new_rule.no_covered_cases))
    f.write('\n-quality: ' + repr(new_rule.quality))
    f.write('\n\n> Number of iterations: ' + repr(idx))
    f.close()

    return new_rule


def rule_pruning(rule, min_case_per_rule, dataset, idx_e, idx_i):
    p_log_file = "log_rule-pruning-fnc.txt"

    f = open(p_log_file, "a+")
    f.write('\n\n\n================== RULE PRUNING LOOP =========================================================================')
    f.write('\n> Stopping condition: pruned rule quality be less than best quality so far or if pruned rule antecedent has just one term')
    f.write('\n> Receives constructed rule > drops each term on antecedent, sequentially from first to last > each term dropped consists on another rule > new pruned rule is the one of higher quality ')
    f.write('\n> IF no new rules have higher quality than the new pruned rule, or if new pruned rule has oly one term in the antecedent > returns pruned rule')
    f.write('\n==============================================================================================================')
    f.write('\n EXTERNAL LOOP ITERATION ' + repr(idx_e))
    f.write('\n INTERNAL LOOP ITERATION ' + repr(idx_i))
    f.write('\n\n> RULE TO BE PRUNED :')
    f.close()
    rule.print_txt(p_log_file, 'Class')
    f = open(p_log_file, "a+")
    f.write('\n-Number of covered cases: ' + repr(rule.no_covered_cases))
    f.write('\n-Quality: ' + repr(rule.quality))
    f.close()

    idx = 0
    new_rule = Rule(dataset)
    improvement = True
    while improvement:
        idx += 1

        improvement = False

        f = open(p_log_file, "a+")
        f.write('\n\n>>>>>>>>>>>>>>> ITERATION ' + repr(idx))
        f.close()

        if len(rule.antecedent) <= 1:
            f = open(p_log_file, "a+")
            f.write('\n\n==> BREAK LOOP:')
            f.write('\n> Condition: pruned rule antecedent = 1')
            # f.write('\n  - Number of iterations: ' + repr(idx))
            f.close()
            break

        f = open(p_log_file, "a+")
        f.write('\n\n==> CURRENT RULE :')
        f.close()
        rule.print_txt(p_log_file, 'Class')
        f = open(p_log_file, "a+")
        f.write('\n\n==> TERMS DROPPING PROCEDURE:')
        f.close()

        antecedent = rule.antecedent.copy()
        best_quality = rule.quality

        for term_idx, attr_drop in enumerate(antecedent):

            f = open(p_log_file, "a+")
            f.write('\n\n>>> TERM ' + repr(term_idx))
            f.write('\n\n> Term_2b_dropped: Attribute=' + repr(attr_drop) + ' Value=' + repr(rule.antecedent[attr_drop]))
            f.close()

            # creates another rule deleting a term from its antecedent
            pruned_rule = Rule(dataset)
            pruned_rule.gen_pruned_rule(rule, attr_drop, term_idx, dataset, min_case_per_rule, idx_e, idx_i)

            f = open(p_log_file, "a+")
            f.write('\n\n> Pruned Rule:')
            f.close()
            pruned_rule.print_txt(p_log_file, 'Class')
            f = open(p_log_file, "a+")
            f.write('\n-Number of covered cases: ' + repr(pruned_rule.no_covered_cases))
            f.write('\n-Quality: ' + repr(pruned_rule.quality))
            f.close()

            if pruned_rule.no_covered_cases < min_case_per_rule:  # POSSIBLE TO HAPPEN?
                f = open(p_log_file, "a+")
                f.write('\n!!!! WARNING: pruned rule covers less cases!')
                f.close()

            if pruned_rule.quality > best_quality:
                new_rule = copy.deepcopy(pruned_rule)
                best_quality = pruned_rule.quality
                improvement = True
                f = open(p_log_file, "a+")
                f.write('\n\n!!! Improvement True')
                f.write('\n!!! Best rule so far')
                f.close()
        # end of for attrs in antecedent

        if improvement:
            rule = copy.deepcopy(new_rule)
    # end of while improvement

    f = open(p_log_file, "a+")
    f.write('\n\n================== END PRUNING FUNCTION LOOP')
    f.write('\n> Condition: best quality of new pruned rules < current rule quality')
    f.write('\n  - Improvement: ' + repr(improvement))
    f.write('\n  - Number of iterations: ' + repr(idx))
    f.write('\n\n> Final Pruned Rule:')
    f.close()
    rule.print_txt(p_log_file, 'Class')

    return rule


def pheromone_updating(list_of_terms, pruned_rule):

    # Getting used terms
    used_terms_idx = []
    for term in pruned_rule.added_terms:
        used_terms_idx.append(term.term_idx)

    # Increasing used terms pheromone
    denominator = 0
    for term in list_of_terms:
        if term.term_idx in used_terms_idx:
            term.pheromone += term.pheromone * pruned_rule.quality
        denominator += term.pheromone

    # Decreasing not used terms: normalization
    for term in list_of_terms:
        term.pheromone = term.pheromone / denominator

    return list_of_terms


def get_remaining_cases_rule(dataset):

    classes = dataset.data[:, dataset.col_index[dataset.class_attr]]
    class_freq = dict(collections.Counter(classes))

    max_freq = 0
    class_chosen = None
    for w in class_freq:                        # other way: class_chosen <= max(class_freq[])
        if class_freq[w] > max_freq:
            class_chosen = w
            max_freq = class_freq[w]

    rule = Rule(dataset)
    rule.covered_cases = []
    rule.consequent = class_chosen

    return rule


def classification_task(dataset, list_of_rules):

    predicted_classes = []
    chosen_class = None
    all_cases = len(dataset.data)
    compatibility = 1

    rules = copy.deepcopy(list_of_rules[:-1])
    remaining_cases_rule = copy.deepcopy(list_of_rules[-1])

    for case in range(all_cases):   # for each new case

        for rule in rules:          # sequential rule compatibility test

            antecedent = rule.antecedent
            for attr in antecedent:
                if antecedent[attr] != dataset.data[case, dataset.col_index[attr]]:
                    compatibility = 0

            if compatibility == 1:
                chosen_class = rule.consequent
                break

        if chosen_class is None:
            chosen_class = remaining_cases_rule.consequent

        predicted_classes.append(chosen_class)
        chosen_class = None

    return predicted_classes






















