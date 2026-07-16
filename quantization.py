import os
import torch
import torch.nn as nn

from model import evaluate_model


def quantize_model(model, test_loader, old_model_path, new_model_path, device):
    """
    Quantize the model by converting all linear layers to dtype of qint8.
    :param model: Model to quantize.
    :param test_loader: DataLoader for the test set.
    :param old_model_path: Place where state of old model is saved.
    :param new_model_path: Place to save the state of new model.
    :param device: Device to use.
    :return: Result of running quantized model on test dataset.
    """
    model.load_state_dict(torch.load(old_model_path, map_location="cpu"))
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save(quantized_model.state_dict(), new_model_path)

    result = evaluate_model(quantized_model, test_loader, device)

    size_bytes = os.path.getsize(new_model_path)
    size_mb = size_bytes / (1024 ** 2)
    result["Size"] = size_mb

    return result