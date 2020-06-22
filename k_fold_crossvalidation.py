from sklearn.model_selection import StratifiedKFold


def k_fold(data, class_attr, k):

    fold = StratifiedKFold(k, shuffle=True)

    training_folders = dict.fromkeys(range(k))
    test_folders = dict.fromkeys(range(k))
    idx = 0

    for train, test in fold.split(data, data[class_attr]):
        training_folders[idx] = list(train)
        test_folders[idx] = list(test)
        idx += 1

    return training_folders, test_folders

