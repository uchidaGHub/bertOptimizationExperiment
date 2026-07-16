import matplotlib.pyplot as plt
import os
import torch
from torch.optim import AdamW
import time

from data import get_agnews_dataloader
from model import create_model, train_model, evaluate_saved_model
from pruning import prune_model_unstructured, first_few_heads, attention_head_pruning, remove_encoder_layers
from quantization import quantize_model


# これはサンプルの Python スクリプトです。

# Shift+F10 を押して実行するか、ご自身のコードに置き換えてください。
# Shift を2回押す を押すと、クラス/ファイル/ツールウィンドウ/アクション/設定を検索します。


def print_hi(name):
    # スクリプトをデバッグするには以下のコード行でブレークポイントを使用してください。
    print(f'Hi, {name}')  # Ctrl+F8を押すとブレークポイントを切り替えます。

def gpu_test(device):
    """
    Test if GPU is working correctly.
    :param device: Device to use.
    """
    print("Running code on GPU...")

    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(100):
        c = a @ b

    torch.cuda.synchronize()
    end = time.time()
    avg = (end - start) / 100

    print("Average time =", avg)

def baseline_bert_exp(training_layers, device, restart=False):
    """
    Experiment which train baseline BERT model.
    :param training_layers: One of (0, 1, 2):
        0: Train all layers.
        1: Train last 4 layers.
        2: Train last layer only.
    :param device: Device to use.
    :param restart: True, then redo all training. False, then resume from checkpoint.
    """
    model_name = "bert-base-uncased"
    # model_name = "distilbert-base-uncased"
    batch_size = 16
    agnews_train_loader = get_agnews_dataloader(model_name, batch_size)
    agnews_test_loader = get_agnews_dataloader(model_name, batch_size, test=True)

    model = create_model(model_name, device)

    if training_layers == 1:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

        for layer in model.bert.encoder.layer[:8]:
            for param in layer.parameters():
                param.requires_grad = False

    elif training_layers == 2:
        for param in model.base_model.parameters():
            param.requires_grad = False

    adam_optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    num_epochs = 3
    model_path = "models/test_model.pth"
    checkpoint_path = "models/checkpoints/test_checkpoint.pth"

    training_result = train_model(
        model, agnews_train_loader, agnews_test_loader, adam_optimizer, num_epochs, model_path, checkpoint_path, device,
        start_over=restart
    )

    print(training_result["best_accuracy"])

def display_result(result):
    """
    Display the model evaluation result contained as dictionary.
    :param result: Model evaluation result contained as dictionary.
    """
    print(f"Accuracy: {result['Accuracy'] * 100:.2f}%")
    print(f"Total Runtime: {result['Total Runtime']:.2f}sec")
    print(f"Latency: {result['Runtime Per Sample'] * 1000:.2f}ms/sample")
    print(f"Throughput: {result['Throughput']:.2f}samples/sec")
    print(f"Size: {result['Size']:.2f}MB")
    print(f"# Parameters: {result['Parameters']}")

def model_eval(model_name, model_path, device):
    """
    Evaluate trained model, and get its accuracy, time, and size.
    :param model_name: Name of the model to load.
    :param model_path: Path which stores model states.
    :param device: Device to use.
    """
    model = create_model(model_name, device)
    agnews_test_loader = get_agnews_dataloader(model_name, 16, test=True)
    result = evaluate_saved_model(model, model_path, agnews_test_loader, device)

    display_result(result)

def bert_quantization():
    """
    Experiment to quantize the model.
    :param device: Device to use.
    """
    model_name = "bert-base-uncased"
    device = torch.device("cpu")
    model = create_model(model_name, device)

    batch_size = 16
    agnews_test_loader = get_agnews_dataloader(model_name, batch_size, test=True)

    old_model_path = "models/test_model.pth"
    new_model_path = "models/quantized_model.pth"

    result = quantize_model(model, agnews_test_loader, old_model_path, new_model_path, device)

    display_result(result)

def unstructured_pruning(device):
    """
    Run unstructured pruning on the trained BERT model.
    :param device: Device to use.
    """
    model_name = "bert-base-uncased"
    model = create_model(model_name, device)

    batch_size = 16
    agnews_test_loader = get_agnews_dataloader(model_name, batch_size, test=True)

    old_model_path = "models/test_model.pth"
    new_model_path = "models/pruned_model_unstructured.pth"

    result = prune_model_unstructured(model, agnews_test_loader, old_model_path, new_model_path, device)

    display_result(result)


def accept_num_input(min_num_allowed, max_num_allowed):
    """
    Accept user input of integers, between two given numbers.
    :param min_num_allowed: Minimum numbers allowed.
    :param max_num_allowed: Maximum numbers allowed.
    :return: Entered number.
    """
    entered_num = min_num_allowed - 1

    while entered_num < min_num_allowed or entered_num > max_num_allowed:
        entered_num = int(input(f"Enter number ({min_num_allowed}-{max_num_allowed}): "))

        if entered_num < min_num_allowed or entered_num > max_num_allowed:
            print("Entered value not valid.")

    return entered_num


def attention_head_pruning_exp(device):
    """
    Run an attention head pruning on BERT model.
    :param device: Device to use.
    """
    model_name = "bert-base-uncased"
    model = create_model(model_name, device)

    print("Select ways to prune heads:")
    print("1. First few of heads from first few layers")
    pruning_method = accept_num_input(1, 1)
    print()

    if pruning_method == 1:
        print("Enter number of layers to remove heads")
        num_layers = accept_num_input(0, 12)
        print()

        print("Enter number of heads to remove for each layer")
        num_heads = accept_num_input(0, 12)
        print()

        heads_to_prune = first_few_heads(num_layers, num_heads)
    else:
        heads_to_prune = dict()

    batch_size = 16
    agnews_test_loader = get_agnews_dataloader(model_name, batch_size, test=True)

    old_model_path = "models/test_model.pth"

    print("Enter the file name to save the pruned model: (do not include .pth)")
    new_path_name = input("File name: models/")
    new_model_path = "models/" + new_path_name + ".pth"

    result = attention_head_pruning(model, heads_to_prune, agnews_test_loader, old_model_path, new_model_path, device)

    display_result(result)

def remove_num_layers(num_layers, device):
    """
    Run a experiment which removes specific numbers of encoder layers.
    :param num_layers: Number of layers to remove.
    :param device: Device to use.
    :return: Model evaluation result.
    """
    model_name = "bert-base-uncased"
    model = create_model(model_name, device)

    batch_size = 16
    agnews_test_loader = get_agnews_dataloader(model_name, batch_size, test=True)

    old_model_path = "models/test_model.pth"
    new_model_path = f"models/pruned_model_{num_layers}layers.pth"

    result = remove_encoder_layers(model, num_layers, agnews_test_loader, old_model_path, new_model_path, device)

    return result

def remove_layers_all_num(device):
    """
    Remove 1, 2, ..., and all encoding layers from BERT model, then plot and save evaluation result.
    :param device: Device to use.
    """
    checkpoint_path = "experiments/checkpoints/encoder_pruning.pth"

    list_checkpoint_attrs = [
        "Accuracy", "Total Runtime", "Runtime Per Sample", "Throughput", "Parameters", "Size"
    ]

    if not os.path.exists(checkpoint_path):
        start_over = True
    else:
        print("Resume from the checkpoint?:")
        print("1. Yes, resume from chackpoint.")
        print("2. No, start over from beginning.")
        start_over = accept_num_input(1, 2) == 2

    if start_over:
        checkpoint = dict()

        for attr in list_checkpoint_attrs:
            checkpoint[attr] = list()
    else:
        checkpoint = torch.load(checkpoint_path)

    while len(checkpoint["Accuracy"]) < 13:
        num_layers = len(checkpoint["Accuracy"])
        print(f"Remove {num_layers}/12 layers...")
        result = remove_num_layers(num_layers, device)

        for attr in list_checkpoint_attrs:
            checkpoint[attr].append(result[attr])

        torch.save(checkpoint, checkpoint_path)

    print("Complete!")
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    x_line = range(13)
    colors = ["red", "blue", "yellow", "green", "orange", "purple"]

    for i, attr in enumerate(list_checkpoint_attrs):
        axs[i // 3, i % 3].plot(x_line, checkpoint[attr], color=colors[i])
        axs[i // 3, i % 3].set_title(attr)
        axs[i // 3, i % 3].set_xlabel("Number of removed layers")

    plt.tight_layout()
    plt.savefig("experiments/figures/encoder_pruning.png")


def remove_layer_pruning(device):
    """
    Run a pruning which removes encoder layers from BERT model.
    :param device: Device to use.
    """
    print("Select an experiment:")
    print("1. Remove specific number of encoder layers.")
    print("2. Remove 1, 2, ..., and all encoding layers, and plot results.")
    exp_num = accept_num_input(1, 2)

    if exp_num == 1:
        print("Enter number of encoder layers to remove:")
        num_layers = accept_num_input(0, 12)
        result = remove_num_layers(num_layers, device)
        display_result(result)
    elif exp_num == 2:
        remove_layers_all_num(device)


# ガター内の緑色のボタンを押すとスクリプトを実行します。
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    print("Select an experiment:")
    print("0. Test GPU.")
    print("1. Baseline BERT training.")
    print("2. Baseline BERT evaluation.")
    print("3. BERT Quantization.")
    print("4. BERT Pruning (unstructured)")
    print("5. BERT Pruning (removing encoder layers)")
    exp_num = accept_num_input(0, 5)
    print()

    if exp_num == 0:
        gpu_test(device)
    elif exp_num == 1:
        print("Which parameters to train?")
        print("1. All parameters (may take REALLY long time).")
        print("2. Parameters on last 4 layers.")
        print("3. Parameters only on classification layer.")
        training_parameters = accept_num_input(1, 3) - 1
        print()

        print("Resume from checkpoint?")
        print("1. Yes, resume from checkpoint.")
        print("2. No, retrain from beginning.")
        restart = accept_num_input(1, 2) == 2
        print()

        baseline_bert_exp(training_parameters, device, restart)
    elif exp_num == 2:
        model_eval("bert-base-uncased", "models/test_model.pth", device)
    elif exp_num == 3:
        bert_quantization()
    elif exp_num == 4:
        unstructured_pruning(device)
    elif exp_num == 5:
        remove_layer_pruning(device)


# PyCharm のヘルプは https://www.jetbrains.com/help/pycharm/ を参照してください
