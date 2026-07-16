import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import transformers

from model import evaluate_model


def prune_model_unstructured(
        model, test_loader, old_model_path, new_model_path, device
):
    """
    Apply unstructured pruning to the given model, and evaluate its performance on the test dataset.
    :param model: Model to be pruned.
    :param test_loader: DataLoder of test dataset.
    :param old_model_path: Path which stores states of the original model.
    :param new_model_path: Path which stores states of the pruned model.
    :param device: Device to use.
    :return: Test evaluation result of the pruned model.
    """
    model.load_state_dict(torch.load(old_model_path))
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.3)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, "weight")

    torch.save(model.state_dict(), new_model_path)
    result = evaluate_model(model, test_loader, device)
    size_bytes = os.path.getsize(new_model_path)
    size_mb = size_bytes / (1024 ** 2)
    result["Size"] = size_mb

    return result

def attention_head_pruning(model, heads_to_prune, test_loader, old_model_path, new_model_path, device):
    """
    Perform attention head pruning on the given model, and evaluate its performance on the test dataset.
    :param model: Model to be pruned.
    :param heads_to_prune: Dictionary of heads to prune.
    :param test_loader: DataLoder of test dataset.
    :param old_model_path: Path which stores states of the original model.
    :param new_model_path: Path which stores states of the pruned model.
    :param device: Device to use.
    :return: Test evaluation result of the pruned model.
    """
    model.load_state_dict(torch.load(old_model_path))
    model.eval()

    for i, layer in enumerate(model.bert.encoder.layer):
        print(type(layer.attention))

    print(dir(model.bert.encoder.layer[0].attention))

    model.bert.prune_heads(heads_to_prune)

    torch.save(model.state_dict(), new_model_path)
    result = evaluate_model(model, test_loader, device)
    size_bytes = os.path.getsize(new_model_path)
    size_mb = size_bytes / (1024 ** 2)
    result["Size"] = size_mb

    return result

def first_few_heads(num_layers, num_heads):
    """
    Get dictionary which represents first few heads of first few layers in the model.
    :param num_layers: Number of layers to remove the heads from.
    :param num_heads: Number of heads to remove for the first few layers.
    :return: Dictionary which contains list of heads to remove.
    """
    heads_to_prune = dict()

    for layer in range(num_layers):
        heads_to_prune[layer] = list(range(num_heads))

    return heads_to_prune

def remove_encoder_layers(model, num_layers, test_loader, old_model_path, new_model_path, device):
    """
    Remove the last few encoder layers from the given model.
    :param model: Model to remove encoder layers from.
    :param num_layers: Number of encoder layers to remove.
    :param test_loader: DataLoder of test dataset.
    :param old_model_path: Path which stores states of the original model.
    :param new_model_path: Path which stores states of the pruned model.
    :param device: Device to use.
    :return: Test evaluation result of the pruned model.
    """
    model.load_state_dict(torch.load(old_model_path))
    model.eval()

    model.bert.encoder.layer = torch.nn.ModuleList(model.bert.encoder.layer[:12 - num_layers])
    model.config.num_hidden_layers = num_layers

    torch.save(model.state_dict(), new_model_path)
    result = evaluate_model(model, test_loader, device)
    size_bytes = os.path.getsize(new_model_path)
    size_mb = size_bytes / (1024 ** 2)
    result["Size"] = size_mb

    return result