import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_average_precision_x_recall_curve_of_all_models_for_a_given_epoch_and_threshold(data, epoch, threshold, colors_model, step = 0.025):
    models_array = []

    for i, model in enumerate(data.keys()):
        models_array.append(model)
        precision_array_all_iter = []
        recall_array_all_iter = []
        recall_thresholds = np.arange(start=0, stop=1+step, step=step)

        precision_a = []
        recall_a = []

        for iter in range(len(data[model])):

            prediction = data[model][str(iter)][str(epoch)]["prediction"][str(threshold)]

            nb_gt = data[model][str(iter)][str(epoch)]["values_computed"][-1]
            acc_TP = 0
            acc_FP = 0
            cnt_detection_reviewed = 0
            precision_array = []
            recall_array = []

            for element in prediction:
                cnt_detection_reviewed += 1
                if element[-1] == True:
                    acc_TP += 1
                else:
                    acc_FP += 1

                precision = acc_TP/cnt_detection_reviewed
                recall = acc_TP/nb_gt

                precision_array.append(precision)
                recall_array.append(recall)
            

            precision_a.append(precision_array)
            recall_a.append(recall_array)
            tmp_array_p = []
            tmp_array_r = []

            for t in recall_thresholds:
                if t <= max(recall_array):
                    tmp_array_p.append(precision_array[min(range(len(recall_array)), key=lambda ele: abs(recall_array[ele]-t))])
                else:
                    tmp_array_p.append(0)
                tmp_array_r.append(t)

            precision_array_all_iter.append(tmp_array_p)
            recall_array_all_iter.append(tmp_array_r)

        

        # print(precision_array_all_iter)

        precision_array_mean = np.array(precision_array_all_iter).mean(axis=0)
        recall_array_mean = np.array(recall_array_all_iter).mean(axis=0)

        # print(recall_array_mean)

        plt.plot(recall_array_mean, precision_array_mean, color=colors_model[i])
    # plt.plot(recall_a[0], precision_a[0])
    plt.legend(models_array)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Average Precision x Recall curve of all model sizes with IOU of : " + str(threshold) + " for epoch : " + str(epoch))
    plt.show()


def plot_average_precision_x_recall_curve(data, model, epoch, threshold):

    precision_array_all_iter = []
    recall_array_all_iter = []
    step = 0.025
    recall_thresholds = np.arange(start=0, stop=1+step, step=step)

    precision_a = []
    recall_a = []

    for iter in range(len(data[model])):

        prediction = data[model][str(iter)][str(epoch)]["prediction"][str(threshold)]

        nb_gt = data[model][str(iter)][str(epoch)]["values_computed"][-1]
        acc_TP = 0
        acc_FP = 0
        cnt_detection_reviewed = 0
        precision_array = []
        recall_array = []

        for element in prediction:
            cnt_detection_reviewed += 1
            if element[-1] == True:
                acc_TP += 1
            else:
                acc_FP += 1

            precision = acc_TP/cnt_detection_reviewed
            recall = acc_TP/nb_gt

            precision_array.append(precision)
            recall_array.append(recall)
        

        precision_a.append(precision_array)
        recall_a.append(recall_array)
        tmp_array_p = []
        tmp_array_r = []

        for t in recall_thresholds:
            if t <= max(recall_array):
                tmp_array_p.append(precision_array[min(range(len(recall_array)), key=lambda ele: abs(recall_array[ele]-t))])
            else:
                tmp_array_p.append(0)
            tmp_array_r.append(t)

        precision_array_all_iter.append(tmp_array_p)
        recall_array_all_iter.append(tmp_array_r)

    

    # print(precision_array_all_iter)

    precision_array_mean = np.array(precision_array_all_iter).mean(axis=0)
    recall_array_mean = np.array(recall_array_all_iter).mean(axis=0)

    # print(recall_array_mean)
    index = 4
    plt.plot(recall_array_mean, precision_array_mean)
    # plt.plot(recall_a[index], precision_a[index])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Average Precision x Recall curve")
    plt.show()


def compute_average_precision_50_95(data_generated, iteration, epoch, compute_only_50=False):
    prediction = data_generated[iteration][epoch]["prediction"]

    ap = []

    for key in prediction:

        nb_gt = data_generated[iteration][epoch]["values_computed"][-1]
        acc_TP = 0
        acc_FP = 0
        cnt_detection_reviewed = 0
        precision_array = []
        recall_array = []

        for element in prediction[key]:
            cnt_detection_reviewed += 1
            if element[-1] == True:
                acc_TP += 1
            else:
                acc_FP += 1

            precision = acc_TP/cnt_detection_reviewed
            recall = acc_TP/nb_gt

            precision_array.append(precision)
            recall_array.append(recall)

        ap.append(np.trapz(precision_array, recall_array))
        if compute_only_50:
            break

    return np.mean(ap)

# model_4_n_16 = np.load("resnet18_cnn_16n_x_y_conf_corrected_final_0_5_05_095.npy", allow_pickle=True).item()

model_4_n_64 = np.load("resnet18_cnn_64n_x_y_conf_corrected_final_0_5_05_095.npy", allow_pickle=True).item()
model_4_n_64_q_8_bits = np.load("qresnet18_cnn_64n_x_y_conf_corrected_final_8_bits_train_from_base_0_5_05_095.npy", allow_pickle=True).item()
model_4_n_64_q__8_bits_fs = np.load("qresnet18_cnn_64n_x_y_conf_corrected_final_8_bits_0_5_05_095.npy", allow_pickle=True).item()
model_4_n_64_q_16_bits_fs = np.load("qresnet18_cnn_64n_x_y_conf_corrected_final_0_5_05_095.npy", allow_pickle=True).item()

# model_4_n_64 = np.load("qresnet18_cnn_64n_x_y_conf_corrected_final_0_5_05_095.npy", allow_pickle=True).item()
# model_4_n_64_8_bits = np.load("qresnet18_cnn_64n_x_y_conf_corrected_final_8_bits_0_5_05_095.npy", allow_pickle=True).item()

dict_of_models = {
    # "model_4_n_16": model_4_n_16,
    "model_4_n_64": model_4_n_64,
    "model_4_n_64_q_8_bits": model_4_n_64_q_8_bits,
    "model_4_n_64_q__8_bits_fs": model_4_n_64_q__8_bits_fs,
    "model_4_n_64_q_16_bits_fs": model_4_n_64_q_16_bits_fs,
}

dict_results = {}

array_epochs = [2, 5, 10, 20, 30, 50, 100, 300, 500, 750, 1000]
array_iou_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
colors_model = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for model in dict_of_models.keys():

    dict_results[model] = {}

    lossess_array = []
    val_lossess_array = []

    f1_scores_array = []
    true_positifs_array = []
    false_positifs_array = []
    false_negatives_array = []

    ap_50_array = []
    ap_50_95_array = []

    for iteration in dict_of_models[model]:
        f1_scores_of_epochs = []
        true_positifs_of_epochs = []
        false_positifs_of_epochs = []
        false_negatives_of_epochs = []
        ap_50_of_epochs = []
        ap_50_95_of_epochs = []
        for epoch in dict_of_models[model][iteration]:
            if epoch not in ['loss', 'val_loss']:
                f1_scores_of_epochs.append(dict_of_models[model][iteration][epoch]['values_computed'][3])
                true_positifs_of_epochs.append(dict_of_models[model][iteration][epoch]['values_computed'][0])
                false_positifs_of_epochs.append(dict_of_models[model][iteration][epoch]['values_computed'][1])
                false_negatives_of_epochs.append(dict_of_models[model][iteration][epoch]['values_computed'][2])
                ap_50_of_epochs.append(compute_average_precision_50_95(dict_of_models[model], iteration, epoch, True))
                ap_50_95_of_epochs.append(compute_average_precision_50_95(dict_of_models[model], iteration, epoch))
            else:
                lossess_array.append(dict_of_models[model][iteration]['loss'])
                val_lossess_array.append(dict_of_models[model][iteration]['val_loss'])

        f1_scores_array.append(f1_scores_of_epochs)
        true_positifs_array.append(true_positifs_of_epochs)
        false_positifs_array.append(false_positifs_of_epochs)
        false_negatives_array.append(false_negatives_of_epochs)
        ap_50_array.append(ap_50_of_epochs)
        ap_50_95_array.append(ap_50_95_of_epochs)

    dict_results[model]["f1_scores_array"] = f1_scores_array
    dict_results[model]["true_positifs_array"] = true_positifs_array
    dict_results[model]["false_positifs_array"] = false_positifs_array
    dict_results[model]["false_negatives_array"] = false_negatives_array
    dict_results[model]["ap_50_array"] = ap_50_array
    dict_results[model]["ap_50_95_array"] = ap_50_95_array
    

# plot_average_precision_x_recall_curve(dict_of_models, "model_xs", 100, 0.5)
plot_average_precision_x_recall_curve_of_all_models_for_a_given_epoch_and_threshold(dict_of_models, 100, 0.5, colors_model)
# for epoch in array_epochs:
#     for t in [array_iou_threshold[0], array_iou_threshold[-1]]:
#         plot_average_precision_x_recall_curve_of_all_models_for_a_given_epoch_and_threshold(dict_of_models, epoch, t, colors_model)


########################################################################################################################
# Pandas output of the data
########################################################################################################################

stats_dict = {}

for model in dict_of_models.keys():

    print("------------------------------------- Displaying model : " + model + " -------------------------------------")

    dict_results[model]["precision_computed"] = [[dict_results[model]["true_positifs_array"][i][j]/max(((dict_results[model]["true_positifs_array"][i][j] + dict_results[model]["false_positifs_array"][i][j]), 1)) for j in range(len(dict_results[model]["true_positifs_array"][i]))]
                        for i in range(len(dict_results[model]["true_positifs_array"]))]

    dict_results[model]["recall_computed"] = [[dict_results[model]["true_positifs_array"][i][j]/max(((dict_results[model]["true_positifs_array"][i][j] + dict_results[model]["false_negatives_array"][i][j]), 1)) for j in range(len(dict_results[model]["false_negatives_array"][i]))]
                        for i in range(len(dict_results[model]["true_positifs_array"]))]

    stats_dict[model] = {
        "epoch": np.array(array_epochs),
        "F1 score mean": np.pad(np.array(dict_results[model]["f1_scores_array"]).mean(axis=0), (0, len(array_epochs)-len(dict_results[model]["f1_scores_array"][0])), constant_values=np.nan),
        "F1 score max": np.pad(np.array(dict_results[model]["f1_scores_array"]).max(axis=0), (0, len(array_epochs)-len(dict_results[model]["f1_scores_array"][0])), constant_values=np.nan),
        "F1 score min": np.pad(np.array(dict_results[model]["f1_scores_array"]).min(axis=0), (0, len(array_epochs)-len(dict_results[model]["f1_scores_array"][0])), constant_values=np.nan),
        "F1 score std": np.pad(np.array(dict_results[model]["f1_scores_array"]).std(axis=0), (0, len(array_epochs)-len(dict_results[model]["f1_scores_array"][0])), constant_values=np.nan),
        "Precision mean": np.pad(np.array(dict_results[model]["precision_computed"]).mean(axis=0), (0, len(array_epochs)-len(dict_results[model]["precision_computed"][0])), constant_values=np.nan),
        "Precision max": np.pad(np.array(dict_results[model]["precision_computed"]).max(axis=0), (0, len(array_epochs)-len(dict_results[model]["precision_computed"][0])), constant_values=np.nan),
        "Precision min": np.pad(np.array(dict_results[model]["precision_computed"]).min(axis=0), (0, len(array_epochs)-len(dict_results[model]["precision_computed"][0])), constant_values=np.nan),
        "Precision std": np.pad(np.array(dict_results[model]["precision_computed"]).std(axis=0), (0, len(array_epochs)-len(dict_results[model]["precision_computed"][0])), constant_values=np.nan),
        "Recall mean": np.pad(np.array(dict_results[model]["recall_computed"]).mean(axis=0), (0, len(array_epochs)-len(dict_results[model]["recall_computed"][0])), constant_values=np.nan),
        "Recall max": np.pad(np.array(dict_results[model]["recall_computed"]).max(axis=0), (0, len(array_epochs)-len(dict_results[model]["recall_computed"][0])), constant_values=np.nan),
        "Recall min": np.pad(np.array(dict_results[model]["recall_computed"]).min(axis=0), (0, len(array_epochs)-len(dict_results[model]["recall_computed"][0])), constant_values=np.nan),
        "Recall std": np.pad(np.array(dict_results[model]["recall_computed"]).std(axis=0), (0, len(array_epochs)-len(dict_results[model]["recall_computed"][0])), constant_values=np.nan),
        "AP[.5] mean": np.pad(np.array(dict_results[model]["ap_50_array"]).mean(axis=0), (0, len(array_epochs)-len(dict_results[model]["ap_50_array"][0])), constant_values=np.nan),
        "AP[.5] max": np.pad(np.array(dict_results[model]["ap_50_array"]).max(axis=0), (0, len(array_epochs)-len(dict_results[model]["ap_50_array"][0])), constant_values=np.nan),
        "AP[.5] min": np.pad(np.array(dict_results[model]["ap_50_array"]).min(axis=0), (0, len(array_epochs)-len(dict_results[model]["ap_50_array"][0])), constant_values=np.nan),
        "AP[.5] std": np.pad(np.array(dict_results[model]["ap_50_array"]).std(axis=0), (0, len(array_epochs)-len(dict_results[model]["ap_50_array"][0])), constant_values=np.nan),
        "AP[.5,0.05,0.95] mean": np.pad(np.array(dict_results[model]["ap_50_95_array"]).mean(axis=0), (0, len(array_epochs)-len(dict_results[model]["ap_50_95_array"][0])), constant_values=np.nan),
        "AP[.5,0.05,0.95] max": np.pad(np.array(dict_results[model]["ap_50_95_array"]).max(axis=0), (0, len(array_epochs)-len(dict_results[model]["ap_50_95_array"][0])), constant_values=np.nan),
        "AP[.5,0.05,0.95] min": np.pad(np.array(dict_results[model]["ap_50_95_array"]).min(axis=0), (0, len(array_epochs)-len(dict_results[model]["ap_50_95_array"][0])), constant_values=np.nan),
        "AP[.5,0.05,0.95] std": np.pad(np.array(dict_results[model]["ap_50_95_array"]).std(axis=0), (0, len(array_epochs)-len(dict_results[model]["ap_50_95_array"][0])), constant_values=np.nan),
        }

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.precision', 4)
    pd.options.display.float_format = '{:.4f}'.format
    df = pd.DataFrame(data=stats_dict[model])

    print("------------------- Precision -------------------")
    for e in range(len(df["epoch"])):
        print(str(df["epoch"][e]) + " & %.4f & %.4f & %.4f & %.4f\\\\" % (df["Precision mean"][e], df["Precision max"][e], df["Precision min"][e], df["Precision std"][e]))
        
    
    print("------------------- Recall -------------------")
    for e in range(len(df["epoch"])):
        print(str(df["epoch"][e]) + " & %.4f & %.4f & %.4f & %.4f\\\\" % (df["Recall mean"][e], df["Recall max"][e], df["Recall min"][e], df["Recall std"][e]))

    print("------------------- F1 score -------------------")
    for e in range(len(df["epoch"])):
        print(str(df["epoch"][e]) + " & %.4f & %.4f & %.4f & %.4f\\\\" % (df["F1 score mean"][e], df["F1 score max"][e], df["F1 score min"][e], df["F1 score std"][e]))

    print("------------------- AP[.5] -------------------")
    for e in range(len(df["epoch"])):
        print(str(df["epoch"][e]) + " & %.4f & %.4f & %.4f & %.4f\\\\" % (df["AP[.5] mean"][e], df["AP[.5] max"][e], df["AP[.5] min"][e], df["AP[.5] std"][e]))

    print("------------------- AP[.5,0.05,0.95] -------------------")
    for e in range(len(df["epoch"])):
        print(str(df["epoch"][e]) + " & %.4f & %.4f & %.4f & %.4f\\\\" % (df["AP[.5,0.05,0.95] mean"][e], df["AP[.5,0.05,0.95] max"][e], df["AP[.5,0.05,0.95] min"][e], df["AP[.5,0.05,0.95] std"][e]))

########################################################################################################################
# Computing and plotting Precision and std over epochs for each model size
########################################################################################################################

f, a = plt.subplots()
a.set_xlabel("Epochs")
a.set_ylabel("Precision")
plt.title("Precision over the epochs for each model size")

p_array = []
model_array = []

for i, model in enumerate(stats_dict.keys()):

    a.fill_between(array_epochs, stats_dict[model]["Precision mean"]-stats_dict[model]["Precision std"], stats_dict[model]["Precision mean"]+stats_dict[model]["Precision std"], alpha=0.3, color=colors_model[i])
    p1 = a.plot(array_epochs, stats_dict[model]["Precision mean"], label=model)
    p2 = a.fill(np.NaN, np.NaN, colors_model[i], alpha=0.3)
    p_array.append((p2[0], p1[0]))
    model_array.append(model)

a.legend(p_array, model_array, loc='lower right')

plt.show()

########################################################################################################################
# Computing and plotting Recall and std over epochs for each model size
########################################################################################################################

f, a = plt.subplots()
a.set_xlabel("Epochs")
a.set_ylabel("Recall")
plt.title("Recall over the epochs for each model size")

p_array = []
model_array = []

for i, model in enumerate(stats_dict.keys()):

    a.fill_between(array_epochs, stats_dict[model]["Recall mean"]-stats_dict[model]["Recall std"], stats_dict[model]["Recall mean"]+stats_dict[model]["Recall std"], alpha=0.3, color=colors_model[i])
    p1 = a.plot(array_epochs, stats_dict[model]["Recall mean"], label=model)
    p2 = a.fill(np.NaN, np.NaN, colors_model[i], alpha=0.3)
    p_array.append((p2[0], p1[0]))
    model_array.append(model)

a.legend(p_array, model_array, loc='lower right')

plt.show()

########################################################################################################################
# Computing and plotting F1 score and std over epochs for each model size
########################################################################################################################

f, a = plt.subplots()
a.set_xlabel("Epochs")
a.set_ylabel("F1 score")
plt.title("F1 score over the epochs for each model size")

p_array = []
model_array = []

for i, model in enumerate(stats_dict.keys()):

    a.fill_between(array_epochs, stats_dict[model]["F1 score mean"]-stats_dict[model]["F1 score std"], stats_dict[model]["F1 score mean"]+stats_dict[model]["F1 score std"], alpha=0.3, color=colors_model[i])
    p1 = a.plot(array_epochs, stats_dict[model]["F1 score mean"], label=model)
    p2 = a.fill(np.NaN, np.NaN, colors_model[i], alpha=0.3)
    p_array.append((p2[0], p1[0]))
    model_array.append(model)

a.legend(p_array, model_array, loc='lower right')

plt.show()

########################################################################################################################
# Computing and plotting AP[.5] and std over epochs for each model size
########################################################################################################################

f, a = plt.subplots()
a.set_xlabel("Epochs")
a.set_ylabel("AP[.5]")
plt.title("AP[.5] over the epochs for each model size")

p_array = []
model_array = []

for i, model in enumerate(stats_dict.keys()):

    a.fill_between(array_epochs, stats_dict[model]["AP[.5] mean"]-stats_dict[model]["AP[.5] std"], stats_dict[model]["AP[.5] mean"]+stats_dict[model]["AP[.5] std"], alpha=0.3, color=colors_model[i])
    p1 = a.plot(array_epochs, stats_dict[model]["AP[.5] mean"], label=model)
    p2 = a.fill(np.NaN, np.NaN, colors_model[i], alpha=0.3)
    p_array.append((p2[0], p1[0]))
    model_array.append(model)

a.legend(p_array, model_array, loc='lower right')

plt.show()

########################################################################################################################
# Computing and plotting AP[.5] and std over epochs for each model size
########################################################################################################################

f, a = plt.subplots()
a.set_xlabel("Epochs")
a.set_ylabel("AP[.5,0.05,0.95]")
plt.title("AP[.5,0.05,0.95] over the epochs for each model size")

p_array = []
model_array = []

for i, model in enumerate(stats_dict.keys()):

    a.fill_between(array_epochs, stats_dict[model]["AP[.5,0.05,0.95] mean"]-stats_dict[model]["AP[.5,0.05,0.95] std"], stats_dict[model]["AP[.5,0.05,0.95] mean"]+stats_dict[model]["AP[.5,0.05,0.95] std"], alpha=0.3, color=colors_model[i])
    p1 = a.plot(array_epochs, stats_dict[model]["AP[.5,0.05,0.95] mean"], label=model)
    p2 = a.fill(np.NaN, np.NaN, colors_model[i], alpha=0.3)
    p_array.append((p2[0], p1[0]))
    model_array.append(model)

a.legend(p_array, model_array, loc='lower right')

plt.show()