import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def read_data(file_path):
    epochs = []
    accuracies = []
    train_loss = []
    val_loss = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header
        for line in file:
            values = line.split()  # Split by whitespace
            if len(values) == 4:  # Ensure the line has the correct number of columns
                epochs.append(int(values[0]))  # First column: Epoch
                train_loss.append(float(values[1]))
                val_loss.append(float(values[2]))
                accuracies.append(float(values[3]))  # Fourth column: Accuracy
    return np.array(epochs), np.array(accuracies), np.array(train_loss), np.array(val_loss)
# Load data from files
sensing_spects_paths = {"Vanilla": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/spects.txt",
              "LoRa (8, 1)": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/spects_lora_(8-1).txt",
              "Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/spects_prefix.txt",
              "ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/spects_elsayedPrefix.txt",
              "LoRa+Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/spects_prefix_lora_(8-1).txt",
              "LoRa+ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/spects_elsayedPrefix_lora_(8-1).txt"} 

sensing_all_paths = {"Vanilla": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/all_small.txt",
              "LoRa (8, 1)": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/all_small_lora_(8-1).txt",
              "Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/all_small_prefix.txt",
              "ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/all_small_elsayedPrefix.txt",
              "LoRa+Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/all_small_prefix_lora_(8-1).txt",
              "LoRa+ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/csi_sensing/all_small_elsayedPrefix_lora_(8-1).txt"}

identification_spects_paths = {"Vanilla": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/spects.txt",
              "LoRa (8, 1)": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/spects_lora_(8-1).txt",
              "Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/spects_prefix.txt",
              "ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/spects_elsayedPrefix.txt",
              "LoRa+Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/spects_prefix_lora_(8-1).txt",
              "LoRa+ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/spects_elsayedPrefix_lora_(8-1).txt"}

identification_all_paths = {"Vanilla": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/small_all.txt",
              "LoRa (8, 1)": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/small_all_lora_(8-1).txt",
              "Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/small_all_prefix.txt",
              "ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/small_all_elsayedPrefix.txt",
              "LoRa+Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/small_all_prefix_lora_(8-1).txt",
              "LoRa+ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/signal_identification/small_all_elsayedPrefix_lora_(8-1).txt"}


positioning_spects_paths = {"Vanilla": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/spects.txt",
              "LoRa (8, 1)": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/spects_lora_(8-1).txt",
              "Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/spects_prefix.txt",
              "ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/spects_elsayedPrefix.txt",
              "LoRa+Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/spects_prefix_lora_(8-1).txt",
              "LoRa+ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/spects_elsayedPrefix_lora_(8-1).txt"} 

positioning_all_paths = {"Vanilla": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/small_all.txt",
              "LoRa (8, 1)": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/small_all_lora_(8-1).txt",
              "Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/small_all_prefix.txt",
              "ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/small_all_elsayedPrefix.txt",
              "LoRa+Prefix": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/small_all_prefix_lora_(8-1).txt",
              "LoRa+ePrefix": "/home/elsayedmohammed/elsayed/finetuning_results2/positioning/small_all_elsayedPrefix_lora_(8-1).txt"} 

def read_accs(data_dict):
    data = {}
    for e, p in data_dict.items():
        epochs, accuracies, _, _ = read_data(p)
        data[e] = accuracies
    return epochs, data

def plot(x, y_dict, to_compare, colors, ytitle='Classification Accuracy (%)', xtitle=None, img_path=None, linestyles=None):
    plt.figure(figsize=(10, 5))

    for i, (exp_label, y_data) in enumerate(y_dict.items()):
        if exp_label in to_compare:
            linestyle = linestyles[i] if linestyles else '-'
            plt.plot(x, y_data, marker='.', linestyle=linestyle, color=colors[exp_label], label=exp_label)
 

    plt.xlabel('Epochs')
    plt.ylabel(ytitle)
    plt.legend()
    plt.grid()
    if xtitle:
        plt.title(xtitle)
    if img_path:
        plt.savefig(img_path)
    plt.close()

if __name__ == '__main__':
    main_img_path = "/home/elsayedmohammed/elsayed/finetuning_results2"
    colors = {"Vanilla": 'black', "LoRa (8, 1)": "green", "Prefix": "red",
              "ePrefix": "grey", "LoRa+Prefix": "brown", "LoRa+ePrefix": "purple"}
    
    epochs, sensing_spects = read_accs(sensing_spects_paths)
    _, sensing_all = read_accs(sensing_all_paths)
    _, positioning_spects = read_accs(positioning_spects_paths)
    _, positioning_all = read_accs(positioning_all_paths)
    _, identification_spects = read_accs(identification_spects_paths)
    _, identification_all = read_accs(identification_all_paths)

    # # 1. Sensing - Spectrograms data
    # xtitle = 'Fine-tuned model accuracy with different finetunig methods\nDataset: Human Activity Sensing\n Model: Pre-trained with the Spectrograms dataset'
    # to_compare = ["Vanilla", "LoRa (8, 1)", "Prefix", "ePrefix", "LoRa+Prefix", "LoRa+ePrefix"]
    # img_path = os.path.join(main_img_path, "csi_sensing/finetune_compare_spects.png")
    # plot(epochs, sensing_spects, to_compare, colors, xtitle=xtitle, img_path=img_path)

    # # 2. Sensing - All data
    # xtitle = 'Fine-tuned model accuracy with different finetunig methods\nDataset: Human Activity Sensing\n Model: Pre-trained with all data'
    # to_compare = ["Vanilla", "LoRa (8, 1)", "Prefix", "ePrefix", "LoRa+Prefix", "LoRa+ePrefix"]
    # img_path = os.path.join(main_img_path, "csi_sensing/finetune_compare_all.png")
    # plot(epochs, sensing_all, to_compare, colors, xtitle=xtitle, img_path=img_path)

    # # 3. Identification - Spectrograms data
    # xtitle = 'Fine-tuned model accuracy with different finetunig methods\nDataset: Signal Identificaiton\n Model: Pre-trained with the Spectrograms dataset'
    # to_compare = ["Vanilla", "LoRa (8, 1)", "Prefix", "ePrefix", "LoRa+Prefix", "LoRa+ePrefix"]
    # img_path = os.path.join(main_img_path, "signal_identification/finetune_compare_spects.png")
    # plot(epochs, identification_spects, to_compare, colors, xtitle=xtitle, img_path=img_path)

    # # 4. Sensing - All data
    # xtitle = 'Fine-tuned model accuracy with different finetunig methods\nDataset: Signal Identificaiton\n Model: Pre-trained with all data'
    # to_compare = ["Vanilla", "LoRa (8, 1)", "Prefix", "ePrefix", "LoRa+Prefix", "LoRa+ePrefix"]
    # img_path = os.path.join(main_img_path, "signal_identification/finetune_compare_all.png")
    # plot(epochs, identification_all, to_compare, colors, xtitle=xtitle, img_path=img_path)

    # # 5. Positioning - Spectrograms data
    # xtitle = 'Fine-tuned model mean positioning error with different finetunig methods\nDataset: Positioning\n Model: Pre-trained with the Spectrograms dataset'
    # ytitle = 'Mean Positioning Error (m)'
    # to_compare = ["Vanilla", "LoRa (8, 1)", "Prefix", "ePrefix", "LoRa+Prefix", "LoRa+ePrefix"]
    # img_path = os.path.join(main_img_path, "positioning/finetune_compare_spects.png")
    # plot(epochs, sensing_spects, to_compare, colors, xtitle=xtitle, img_path=img_path, ytitle=ytitle)

    # # 6. Positioning - All data
    # xtitle = 'Fine-tuned model mean positioning error with different finetunig methods\nDataset: Positioning\n Model: Pre-trained with all data'
    # ytitle = 'Mean Positioning Error (m)'
    # to_compare = ["Vanilla", "LoRa (8, 1)", "Prefix", "ePrefix", "LoRa+Prefix", "LoRa+ePrefix"]
    # img_path = os.path.join(main_img_path, "positioning/finetune_compare_all.png")
    # plot(epochs, sensing_all, to_compare, colors, xtitle=xtitle, img_path=img_path, ytitle=ytitle)

    # 7. Sensing (all vs spectrogram) (Vanilla x LoRa x ePreix+LoRA)
    linestyles = ['-', '--', '-', '--', '-', '--']
    xtitle = 'Fine-tuned model accuracy with different finetunig methods and data\nDataset: Human Activity Sensing\n Model: Pre-trained with spectrograms vs all data'
    img_path = os.path.join(main_img_path, "csi_sensing/compare_all_vs_spects.png")
    colors2 = {"Vanilla (spects)": 'black', "Vanilla (all)": 'black',
               "LoRa (8, 1) (spects)": "green", "LoRa (8, 1) (all)": "green",
               "Prefix (spects)": "red", "Prefix (all)": "red",
               "ePrefix (spects)": "grey", "ePrefix (all)": "grey",
               "LoRa+Prefix (spects)": "brown", "LoRa+Prefix (all)": "brown",
               "LoRa+ePrefix (spects)": "purple", "LoRa+ePrefix (all)": "purple"}
    to_compare = ["Vanilla (spects)", "Vanilla (all)", "LoRa (8, 1) (spects)", "LoRa (8, 1) (all)", 
                  "LoRa+ePrefix (spects)", "LoRa+ePrefix (all)"]
    sensing_custom_data = {"Vanilla (spects)": sensing_spects["Vanilla"],
                           "Vanilla (all)": sensing_all["Vanilla"],
                           "LoRa (8, 1) (spects)": sensing_spects["LoRa (8, 1)"],
                           "LoRa (8, 1) (all)": sensing_all["LoRa (8, 1)"],
                           "LoRa+ePrefix (spects)": sensing_spects["LoRa+ePrefix"],
                           "LoRa+ePrefix (all)": sensing_all["LoRa+ePrefix"]}
    plot(epochs, sensing_custom_data, to_compare, colors2, xtitle=xtitle, img_path=img_path, linestyles=linestyles)

    # 8. Identification (all vs spectrogram) (Vanilla x LoRa x ePreix+LoRA)
    linestyles = ['-', '--', '-', '--', '-', '--']
    xtitle = 'Fine-tuned model accuracy with different finetunig methods and data\nDataset: Signal Identification\n Model: Pre-trained with spectrograms vs all data'
    img_path = os.path.join(main_img_path, "signal_identification/compare_all_vs_spects.png")
    identification_custom_data = {"Vanilla (spects)": identification_spects["Vanilla"],
                           "Vanilla (all)": identification_all["Vanilla"],
                           "LoRa (8, 1) (spects)": identification_spects["LoRa (8, 1)"],
                           "LoRa (8, 1) (all)": identification_all["LoRa (8, 1)"],
                           "LoRa+ePrefix (spects)": identification_spects["LoRa+ePrefix"],
                           "LoRa+ePrefix (all)": identification_all["LoRa+ePrefix"]}
    plot(epochs, identification_custom_data, to_compare, colors2, xtitle=xtitle, img_path=img_path, linestyles=linestyles)

    # 9. Positioning (all vs spectrogram) (Vanilla x LoRa x ePreix+LoRA)
    linestyles = ['-', '--', '-', '--', '-', '--']
    xtitle = 'Fine-tuned model mean positioning error with different finetunig methods and data\nDataset: Positioning\n Model: Pre-trained with spectrograms vs all data'
    ytitle = 'Mean Positioning Error (m)'
    img_path = os.path.join(main_img_path, "positioning/compare_all_vs_spects.png")
    positioning_custom_data = {"Vanilla (spects)": positioning_spects["Vanilla"],
                           "Vanilla (all)": positioning_all["Vanilla"],
                           "LoRa (8, 1) (spects)": positioning_spects["LoRa (8, 1)"],
                           "LoRa (8, 1) (all)": positioning_all["LoRa (8, 1)"],
                           "LoRa+ePrefix (spects)": positioning_spects["LoRa+ePrefix"],
                           "LoRa+ePrefix (all)": positioning_all["LoRa+ePrefix"]}
    plot(epochs, positioning_custom_data, to_compare, colors2, xtitle=xtitle, ytitle=ytitle, img_path=img_path, linestyles=linestyles)


