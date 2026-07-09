from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def get_agnews_dataloader(model_name, batch_size, test=False, num_workers=0):
    """
    Get DataLoader object of AG News dataset.
    :param model_name: Name of the model. Used for tokenization.
    :param batch_size: Batch size.
    :param test: If true, then get test data. If not, get training data.
    :param num_workers: Number of workers used in data loading.
    :return: DataLoader object of AG News dataset.
    """
    dataset = load_dataset("wangrongsheng/ag_news")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(sample):
        """
        Tokenize given AG News data.
        :param sample: One sample of AG News dataset.
        :return: Tokenized AG News data.
        """
        tokenized_sample = tokenizer(sample["text"], padding="max_length", truncation=True, max_length=128)

        return tokenized_sample

    tokenized_dataset = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    if test:
        split = "test"
    else:
        split = "train"

    dataset_split = tokenized_dataset[split]

    dataloader = DataLoader(
        dataset_split, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(num_workers != 0)
    )

    return dataloader