from transformers import AutoModelForSequenceClassification
import time
import torch
import torch.nn as nn
import os

def create_model(model_name, device):
    """
    Load the model and move it to device.
    :param model_name: Name of the model to load.
    :param device: Device to use.
    :return: Model object created.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    model.to(device)

    return model

def train_model_epoch(
        model, train_loader, optimizer, checkpoint, checkpoint_path, device, scheduler
):
    """
    Train the model for one epoch.
    :param model: Model to be trained.
    :param train_loader: DataLoader for the training set.
    :param optimizer: Optimizer algorithm.
    :param checkpoint: Checkpoint to save for each 1000 batches.
    :param checkpoint_path: Path to save the checkpoint.
    :param device: Device to use.
    :param scheduler: Learning rate scheduler.
    :return: Average loss for this epoch and training accuracy of the model after this epoch.
    """
    model.train()

    batch_idx = checkpoint["batch_idx"]
    running_loss = torch.tensor(checkpoint["running_loss"], device=device)
    num_correct = torch.tensor(checkpoint["num_correct"], device=device, dtype=torch.long)
    num_samples = torch.tensor(checkpoint["num_samples"], device=device, dtype=torch.long)

    train_itr = iter(train_loader)

    for _ in range(batch_idx):
        next(train_itr)

    while True:
        try:
            batch = next(train_itr)
        except StopIteration:
            break

        if batch_idx % 100 == 99:
            print(f"    Training batch {batch_idx + 1}/{len(train_loader)}...")

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_idx += 1
        running_loss += loss.detach()
        num_correct += (pred == labels).sum()
        num_samples += labels.size(0)

        if batch_idx % 1000 == 0:
            checkpoint["batch_idx"] = batch_idx
            checkpoint["running_loss"] = running_loss.item()
            checkpoint["num_correct"] = num_correct.item()
            checkpoint["num_samples"] = num_samples.item()

            torch.save(checkpoint, checkpoint_path)
            print("--- Progress saved ---")

    avg_loss = (running_loss / len(train_loader)).item()
    accuracy = (num_correct.float() / num_samples.float()).item()

    return avg_loss, accuracy

def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model.
    :param model: Model to be evaluated.
    :param test_loader: DataLoader for the test set.
    :param device: Device to use.
    :return: Average loss and test accuracy of the model.
    """
    model.eval()
    total_loss = 0
    num_correct = 0
    num_samples = 0
    total_time = 0.0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            torch.cuda.synchronize()
            start = time.perf_counter()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            torch.cuda.synchronize()
            end = time.perf_counter()

            loss = outputs.loss
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            num_correct += (pred == labels).sum().item()
            num_samples += labels.size(0)
            total_time += end - start

    num_parameters = sum(p.numel() for p in model.parameters())

    result = {
        "Average Loss": total_loss / len(test_loader),
        "Accuracy": num_correct / num_samples,
        "Total Runtime": total_time,
        "Runtime Per Sample": total_time / num_samples,
        "Throughput": num_samples / total_time,
        "Parameters": num_parameters
    }

    return result

def train_model(
        model, train_loader, test_loader, optimizer, num_epochs,
        model_path, checkpoint_path, device, scheduler=None, start_over=False, # checkpoint_every=0
    ):
    """
    Train the model for one epoch.
    :param model: Model to be trained.
    :param train_loader: DataLoader for the training set.
    :param test_loader: DataLoader for the test set.
    :param optimizer: Optimizer algorithm.
    :param device: Device to use.
    :param num_epochs: Number of epochs to train.
    :param model_path: Name of the path to save the model.
    :param checkpoint_path: Name of the path to save the checkpoint.
    :param scheduler: Learning rate scheduler.
    :param start_over: True, then start training from beginning.
    :param checkpoint_every: Number of batches to train before saving.
    :return: Best accuracy score of all epochs, and list of train and test accuracy scores and loss values over all epochs.
    """
    if not os.path.exists(checkpoint_path):
        start_over = True

    if start_over:
        best_accuracy = 0.0
        epoch = 0
        batch_idx = 0
        running_loss = 0.0
        num_correct = 0
        num_samples = 0
        list_train_loss = list()
        list_train_accuracy = list()
        list_test_loss = list()
        list_test_accuracy = list()
        best_model_state = dict()
    else:
        loaded_checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(loaded_checkpoint["current_model_state"])
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])
        epoch = loaded_checkpoint["epoch"]
        batch_idx = loaded_checkpoint["batch_idx"]
        running_loss = loaded_checkpoint["running_loss"]
        num_correct = loaded_checkpoint["num_correct"]
        num_samples = loaded_checkpoint["num_samples"]
        best_accuracy = loaded_checkpoint["best_accuracy"]
        list_train_loss = loaded_checkpoint["list_train_loss"]
        list_train_accuracy = loaded_checkpoint["list_train_accuracy"]
        list_test_loss = loaded_checkpoint["list_test_loss"]
        list_test_accuracy = loaded_checkpoint["list_test_accuracy"]
        best_model_state = loaded_checkpoint["best_model_state"]

    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "running_loss": running_loss,
        "num_correct": num_correct,
        "num_samples": num_samples,
        "best_accuracy": best_accuracy,
        "current_model_state": model.state_dict(),
        "best_model_state": best_model_state,
        "optimizer_state": optimizer.state_dict(),
        "list_train_loss": list_train_loss,
        "list_train_accuracy": list_train_accuracy,
        "list_test_loss": list_test_loss,
        "list_test_accuracy": list_test_accuracy
    }

    while epoch < num_epochs:
        print(f"Training Epoch: {epoch + 1}/{num_epochs}...")

        train_loss, train_accuracy = train_model_epoch(
            model, train_loader, optimizer, checkpoint, checkpoint_path, device, scheduler
        )

        result = evaluate_model(model, test_loader, device)
        test_loss = result["Average Loss"]
        test_accuracy = result["Accuracy"]

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict()

        list_train_loss.append(train_loss)
        list_train_accuracy.append(train_accuracy)
        list_test_loss.append(test_loss)
        list_test_accuracy.append(test_accuracy)
        epoch += 1

        checkpoint = {
            "epoch": epoch,
            "batch_idx": 0,
            "running_loss": 0.0,
            "num_correct": 0,
            "num_samples": 0,
            "best_accuracy": best_accuracy,
            "current_model_state": model.state_dict(),
            "best_model_state": best_model_state,
            "optimizer_state": optimizer.state_dict(),
            "list_train_loss": list_train_loss,
            "list_train_accuracy": list_train_accuracy,
            "list_test_loss": list_test_loss,
            "list_test_accuracy": list_test_accuracy
        }

        torch.save(checkpoint, checkpoint_path)
        print("--- Progress saved ---")

    torch.save(best_model_state, model_path)
    print("Training Complete!")

    training_result = {
        "best_accuracy": best_accuracy,
        "list_train_loss": list_train_loss,
        "list_train_accuracy": list_train_accuracy,
        "list_test_loss": list_test_loss,
        "list_test_accuracy": list_test_accuracy
    }

    return training_result

def evaluate_saved_model(model, model_path, test_loader, device):
    """
    Load model from path, and evaluate its test accuracy.
    :param model: Model to be evaluated.
    :param model_path: Path to the model state.
    :param test_loader: DataLoader of test dataset.
    :param device: Device to use.
    :return: Average loss and Accuracy score.
    """
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("WARNING: Model of given path does not exist.")

    result = evaluate_model(model, test_loader, device)

    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 ** 2)
    result["Size"] = size_mb

    return result

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

