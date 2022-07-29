import copy
import csv
import json
from enum import Enum
from pathlib import Path
import numpy as np
import torch
from scipy import optimize
from tqdm import tqdm

from exp1_5.config.parser import Configuration

class NetworkTypes(Enum):
    BINARY_ENSAMBLE = 0
    MULTICLASS = 1
    BINARY_ENSAMBLE_M = 2
    BINARY_ENSAMBLE_B = 3
    MULTICLASS_ENSAMBLE = 4

    @staticmethod
    def get_network_type(network_type: str):
        if network_type == 'BE':
            return NetworkTypes.BINARY_ENSAMBLE
        elif network_type == 'MC':
            return NetworkTypes.MULTICLASS
        elif network_type == 'BEM':
            return NetworkTypes.BINARY_ENSAMBLE_M
        elif network_type == 'BEB':
            return NetworkTypes.BINARY_ENSAMBLE_B
        elif network_type == 'MCE':
            return NetworkTypes.MULTICLASS_ENSAMBLE
        else:
            raise ValueError('Network type {} not recognized'.format(network_type))

def error_th_mce(thresholds, num_classes, probabilities, prediction, labels, total_items, classes_per_task):
    predicted = copy.deepcopy(prediction)
    initial_probabilities = copy.deepcopy(probabilities)
    for classifier_idx in range(num_classes // classes_per_task):
        output_start_idx = classifier_idx * classes_per_task
        output_end_idx = output_start_idx + classes_per_task
        classifier_prob = initial_probabilities[output_start_idx:output_end_idx, :]
        classifier_th = thresholds[output_start_idx:output_end_idx]
        classifier_override = classifier_prob >= classifier_th[:, None]
        image_with_override = np.any(classifier_override, 0)
        classifier_prob[~classifier_override] = 0
        rev_c_prob = classifier_override[::-1]
        classifier_best_class = output_start_idx + len(rev_c_prob) - np.argmax(rev_c_prob, 0) - 1
        predicted[image_with_override] = classifier_best_class[image_with_override]

    error = 1 - (predicted == labels).sum() / total_items
    return error.item()


def apply_th_mce(thresholds, num_classes, temp_prediction, probabilities, labels, classes_per_task):
    th_conf_mat = np.zeros((num_classes, num_classes))
    total_items = len(labels)
    if thresholds is not None:
        predicted = copy.deepcopy(temp_prediction)
        initial_probabilities = copy.deepcopy(probabilities)
        for classifier_idx in range(num_classes // classes_per_task):
            output_start_idx = classifier_idx * classes_per_task
            output_end_idx = output_start_idx + classes_per_task
            classifier_prob = initial_probabilities[output_start_idx:output_end_idx, :]
            classifier_th = thresholds[output_start_idx:output_end_idx]
            classifier_override = classifier_prob >= classifier_th[:, None]
            image_with_override = np.any(classifier_override, 0)
            classifier_prob[~classifier_override] = 0
            rev_c_prob = classifier_override[::-1]
            classifier_best_class = output_start_idx + len(rev_c_prob) - np.argmax(rev_c_prob, 0) - 1
            predicted[image_with_override] = classifier_best_class[image_with_override]

    results_mask = (predicted == labels)
    correct_pred_per_class = np.zeros(num_classes)
    total_items_per_class = np.zeros(num_classes)

    correct_unique_labels, correct_counts = np.unique(labels[results_mask], return_counts=True)
    correct_pred_per_class[correct_unique_labels] += correct_counts

    unique_labels, counts = np.unique(labels, return_counts=True)
    total_items_per_class[unique_labels] += counts

    per_class_accuracy = 100 * correct_pred_per_class / total_items_per_class
    avg_accuracy = (correct_pred_per_class / total_items).sum() * 100

    np.add.at(th_conf_mat, (labels, predicted), 1)

    results_dict = {
        "Threshold": thresholds,
        "Accuracy": avg_accuracy,
        "Accuracy per class": per_class_accuracy.tolist(),
        "Confusion matrix": th_conf_mat.tolist()
    }

    return results_dict


def error_th_mce_smart(thresholds, num_classes, probabilities, prediction, labels, total_items, classes_per_task):
    predicted = copy.deepcopy(prediction)
    initial_probabilities = copy.deepcopy(probabilities)
    for classifier_idx in range(num_classes // classes_per_task):
        output_start_idx = classifier_idx * classes_per_task
        output_end_idx = output_start_idx + classes_per_task
        classifier_prob = initial_probabilities[output_start_idx:output_end_idx, :]
        classifier_th = thresholds[output_start_idx:output_end_idx]
        classifier_override = classifier_prob >= classifier_th[:, None]
        image_with_override = np.any(classifier_override, 0)
        classifier_prob[~classifier_override] = 0
        classifier_best_class = output_start_idx + classifier_prob.argmax(0)
        predicted[image_with_override] = classifier_best_class[image_with_override]

    error = 1 - (predicted == labels).sum() / total_items
    return error.item()


def apply_th_mce_smart(thresholds, num_classes, temp_prediction, probabilities, labels, classes_per_task):
    th_conf_mat = np.zeros((num_classes, num_classes))
    total_items = len(labels)
    if thresholds is not None:
        predicted = copy.deepcopy(temp_prediction)
        initial_probabilities = copy.deepcopy(probabilities)
        for classifier_idx in range(num_classes // classes_per_task):
            output_start_idx = classifier_idx * classes_per_task
            output_end_idx = output_start_idx + classes_per_task
            classifier_prob = initial_probabilities[output_start_idx:output_end_idx, :]
            classifier_th = thresholds[output_start_idx:output_end_idx]
            classifier_override = classifier_prob >= classifier_th[:, None]
            image_with_override = np.any(classifier_override, 0)
            classifier_prob[~classifier_override] = 0
            classifier_best_class = output_start_idx + classifier_prob.argmax(0)
            predicted[image_with_override] = classifier_best_class[image_with_override]

    results_mask = (predicted == labels)
    correct_pred_per_class = np.zeros(num_classes)
    total_items_per_class = np.zeros(num_classes)

    correct_unique_labels, correct_counts = np.unique(labels[results_mask], return_counts=True)
    correct_pred_per_class[correct_unique_labels] += correct_counts

    unique_labels, counts = np.unique(labels, return_counts=True)
    total_items_per_class[unique_labels] += counts

    per_class_accuracy = 100 * correct_pred_per_class / total_items_per_class
    avg_accuracy = (correct_pred_per_class / total_items).sum() * 100

    np.add.at(th_conf_mat, (labels, predicted), 1)

    results_dict = {
        "Threshold": thresholds,
        "Accuracy": avg_accuracy,
        "Accuracy per class": per_class_accuracy.tolist(),
        "Confusion matrix": th_conf_mat.tolist()
    }

    return results_dict


def error_th(thresholds, num_classes, probabilities, prediction, labels, total_items):
    predicted = copy.deepcopy(prediction)
    for i_class in range(num_classes):
        idx = probabilities[i_class, :] >= thresholds[i_class]
        predicted[idx] = i_class

    error = 1 - (predicted == labels).sum() / total_items
    return error.item()


def apply_th(threshold, num_classes, temp_prediction, probabilities, labels):
    th_conf_mat = np.zeros((num_classes, num_classes))
    total_items = len(labels)
    if threshold is not None:
        for i_class in range(num_classes):
            if type(threshold) is list or type(threshold) is np.ndarray:
                idx = probabilities[i_class, :] >= threshold[i_class]
            else:
                idx = probabilities[i_class, :] >= threshold

            temp_prediction[idx] = i_class

    results_mask = (temp_prediction == labels)
    correct_pred_per_class = np.zeros(num_classes)
    total_items_per_class = np.zeros(num_classes)

    correct_unique_labels, correct_counts = np.unique(labels[results_mask], return_counts=True)
    correct_pred_per_class[correct_unique_labels] += correct_counts

    unique_labels, counts = np.unique(labels, return_counts=True)
    total_items_per_class[unique_labels] += counts

    per_class_accuracy = 100 * correct_pred_per_class / total_items_per_class
    avg_accuracy = (correct_pred_per_class / total_items).sum() * 100

    np.add.at(th_conf_mat, (labels, temp_prediction), 1)

    results_dict = {
        "Threshold": threshold,
        "Accuracy": avg_accuracy,
        "Accuracy per class": per_class_accuracy.tolist(),
        "Confusion matrix": th_conf_mat.tolist()
    }

    return results_dict


def list_to_p(elements):
    return " ".join(str(elem)[0:7] for elem in elements)


def calc_mce(file_content, smart_override, classes_per_task=10):
    classes_per_task_list = type(classes_per_task) is list
    train = file_content['train_metrics']
    test = file_content['test_metrics']
    exemplar_idx_per_task = file_content['inc_train_exemplar_idx']
    exemplar_idx_per_task = [{int(float(k)): v for k, v in exemplars.items()} for exemplars in exemplar_idx_per_task]
    train_labels = np.array(train['labels'])
    test_labels = np.array(test['labels'])
    test_accuracies = []
    train_accuracies = []
    test_th_accuracies = []
    train_th_accuracies = []

    error_func = error_th_mce_smart if smart_override else error_th_mce
    apply_th_func = apply_th_mce_smart if smart_override else apply_th_mce

    train_prob = np.exp(np.array(train['predictions']))
    test_prob = np.exp(np.array(test['predictions']))

    num_classifiers = len(classes_per_task) if classes_per_task_list else int(train_prob.shape[0] / classes_per_task)
    if classes_per_task_list:
        class_intervals = [0] + classes_per_task
    for i in tqdm(range(0, num_classifiers)):
        if classes_per_task_list:
            current_classes = classes_per_task[i]
            num_classes = sum(class_intervals[:i + 2])
            current_end_class = sum(class_intervals[:i + 2])
        else:
            current_classes = classes_per_task
            num_classes = (i + 1) * current_classes
            current_end_class = (i + 1) * current_classes
        k = i if i == 0 else i + 1

        all_task_ex_idx = sum([exemplar_idx_per_task[k][exemplar_idx]
                               for exemplar_idx in range(0, current_end_class)], [])

        all_task_prob = train_prob[range(0, current_end_class)][:, all_task_ex_idx]
        all_task_pred = all_task_prob.argmax(0)
        all_task_true = train_labels[all_task_ex_idx]
        total_items = all_task_true.shape[0]
        train_ag_acc = 100 * (all_task_pred == all_task_true).sum() / total_items

        temp_predictions = copy.deepcopy(all_task_pred)
        thresholds = np.array([0.5] * num_classes)
        result = optimize.minimize(error_func, thresholds,
                                   (num_classes, all_task_prob, temp_predictions, all_task_true, total_items,
                                    current_classes),
                                   method='Powell',  # method='Nelder-Mead',
                                   bounds=[(0.5, 1.0) for _ in range(num_classes)],
                                   tol=1e-6,
                                   options=dict(disp=False, maxiter=1000000))
        thresholds = result.x
        results_dict = apply_th_func(thresholds, num_classes, all_task_pred, all_task_prob, all_task_true,
                                     current_classes)
        train_th_acc = results_dict['Accuracy']

        test_data_idx = np.isin(test_labels, (range(0, current_end_class)))
        all_test_prob = test_prob[range(0, current_end_class)][:, test_data_idx]
        all_test_pred = all_test_prob.argmax(0)
        all_test_true = test_labels[test_data_idx]
        total_items = all_test_true.shape[0]
        test_ag_acc = 100 * (all_test_pred == all_test_true).sum() / total_items

        results_dict = apply_th_func(thresholds, num_classes, all_test_pred, all_test_prob, all_test_true,
                                     current_classes)

        test_th_acc = results_dict['Accuracy']
        train_accuracies.append(train_ag_acc)
        train_th_accuracies.append(train_th_acc)
        test_accuracies.append(test_ag_acc)
        test_th_accuracies.append(test_th_acc)

    return test_th_accuracies  # acc with th at every step


def calc_b(file_content, classes_per_task=10):
    train = file_content['train_metrics']
    test = file_content['test_metrics']
    exemplar_idx_per_task = file_content['inc_train_exemplar_idx']
    train_prob = np.array(train['predictions'])
    train_labels = np.array(train['labels'])
    test_prob = np.array(test['predictions'])
    test_labels = np.array(test['labels'])
    test_accuracies = []
    train_accuracies = []
    test_th_accuracies = []
    train_th_accuracies = []
    test_no_opt_th_accuracies = []

    num_classes = train_prob.shape[0]
    current_end_class = num_classes

    if len(exemplar_idx_per_task) > 0:
        all_task_ex_idx = sum([exemplar_idx_per_task[-1][str(exemplar_idx)]
                               for exemplar_idx in range(0, current_end_class)], [])
    else:
        all_task_ex_idx = np.where(np.isin(train_labels, range(0, current_end_class)))[0]

    all_task_prob = train_prob[range(0, current_end_class)][:, all_task_ex_idx]
    all_task_pred = all_task_prob.argmax(0)
    all_task_true = train_labels[all_task_ex_idx]
    total_items = all_task_true.shape[0]
    train_ag_acc = 100 * (all_task_pred == all_task_true).sum() / total_items

    temp_predictions = copy.deepcopy(all_task_pred)
    thresholds = np.array([0.5] * num_classes)
    result = optimize.minimize(error_th, thresholds,
                               (num_classes, all_task_prob, temp_predictions, all_task_true, total_items),
                               method='Powell',  # method='Nelder-Mead',
                               bounds=[(0.5, 1.0) for _ in range(num_classes)],
                               tol=1e-6,
                               options=dict(disp=False, maxiter=1000000))
    thresholds = result.x
    results_dict = apply_th(thresholds, num_classes, all_task_pred, all_task_prob, all_task_true)
    train_th_acc = results_dict['Accuracy']

    test_data_idx = np.isin(test_labels, (range(0, current_end_class)))
    all_test_prob = test_prob[range(0, current_end_class)][:, test_data_idx]
    all_test_pred = all_test_prob.argmax(0)
    all_test_true = test_labels[test_data_idx]
    total_items = all_test_true.shape[0]
    test_ag_acc = 100 * (all_test_pred == all_test_true).sum() / total_items

    results_dict = apply_th(thresholds, num_classes, copy.deepcopy(all_test_pred), all_test_prob,
                            all_test_true)  # th optimization
    results_dict_no_opt_th = apply_th([0.5] * len(thresholds), num_classes, copy.deepcopy(all_test_pred),
                                      all_test_prob, all_test_true)  # no th optimization

    test_th_acc = results_dict['Accuracy']
    test_no_opt_th_acc = results_dict_no_opt_th['Accuracy']
    train_accuracies.append(train_ag_acc)
    train_th_accuracies.append(train_th_acc)
    test_accuracies.append(test_ag_acc)
    test_th_accuracies.append(test_th_acc)
    test_no_opt_th_accuracies.append(test_no_opt_th_acc)

    return test_th_accuracies, test_no_opt_th_accuracies, test_accuracies  # acc with th at every step


def calc_mc(file_content, key='test_metrics'):
    test = file_content[key]
    test_prob = np.exp(np.array(test['predictions']))
    test_true = np.array(test['labels'])
    test_pred = test_prob.argmax(0)
    total_items = test_true.shape[0]
    test_ag_acc = 100 * (test_pred == test_true).sum() / total_items
    return test_ag_acc


def main(search_folder_path, run_names, csv_file_path, smart_override, incremental_setup=None):
    Path(csv_file_path).parent.mkdir(exist_ok=True, parents=True)
    csv_file = open(csv_file_path, 'w')
    csv_writer = csv.writer(csv_file)

    if incremental_setup is None:
        incremental_setup = list(range(10, 101, 10))

    csv_writer.writerow(
        ['Run', 'Time', 'Final Acc', 'Inc acc'] + incremental_setup + incremental_setup)

    for run_name in run_names:
        inc_mean_acc_list = []
        final_acc_list = []
        time_list = []
        inc_acc_list = []
        if (search_folder_path / run_name).exists():
            for i, folder in enumerate([Path(search_folder_path / run_name / str(idx)) for idx in range(0, 10)]):
                if not folder.exists():
                    print('Missing run {} from {}, please rerun the missing experiment'.format(i, run_name))
                    continue

                print("Parsing : {}".format(folder))

                json_path = folder / 'results_test.json'
                config = Configuration(json_path.parent / 'config.json')
                network_type = NetworkTypes.get_network_type(config.get_param_value('network_type'))
                dataset_name = config.get_param_value('dataset_name')

                fp = open(str(json_path), 'r')
                json_content = json.load(fp)
                fp.close()

                if network_type == NetworkTypes.MULTICLASS:
                    final_acc = calc_mc(json_content)
                    final_acc_list.append(final_acc)
                    time_list.append(json_content['time']['train'])  # total
                elif network_type == NetworkTypes.BINARY_ENSAMBLE_B:
                    if config.get_param_value('threshold', mandatory=False) == 'default':
                        _, test_accuracies, _ = calc_b(json_content, 1)  # without optmizied TH (th=0.5)
                    else:
                        test_accuracies, _, _ = calc_b(json_content, 1)  # with optimized TH

                    final_acc_list.append(test_accuracies[-1])
                    time_list.append(json_content['time']['train'])
                elif network_type == NetworkTypes.MULTICLASS_ENSAMBLE:
                    classes_per_group = config.get_param_value('num_classes_per_group')
                    test_accuracies = calc_mce(json_content, smart_override, classes_per_group)

                    if dataset_name == "CUB200":
                        inc_mean_acc_list.append(np.array(test_accuracies)[1:].mean())
                    else:
                        inc_mean_acc_list.append(np.array(test_accuracies).mean())

                    final_acc_list.append(test_accuracies[-1])
                    inc_acc_list.append(test_accuracies)
                    time_list.append(json_content['time']['train'])

            if network_type == NetworkTypes.MULTICLASS or network_type == NetworkTypes.BINARY_ENSAMBLE_B:
                csv_writer.writerow([run_name, np.array(time_list).mean(), np.array(final_acc_list).mean()])
            elif network_type == NetworkTypes.MULTICLASS_ENSAMBLE:
                csv_writer.writerow([run_name, np.array(time_list).mean(), np.array(final_acc_list).mean(),
                                     np.array(inc_mean_acc_list).mean()] +
                                    np.array(inc_acc_list).mean(0).tolist() + np.array(inc_acc_list).std(0).tolist())
            csv_file.flush()
        else:
            print("Run '{}' not found, results won't be complete".format(run_name))
    csv_file.close()
