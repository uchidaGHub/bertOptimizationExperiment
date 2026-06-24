import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import datasets
import huggingface_hub
import transformers

from data import get_agnews_dataloader


# これはサンプルの Python スクリプトです。

# Shift+F10 を押して実行するか、ご自身のコードに置き換えてください。
# Shift を2回押す を押すと、クラス/ファイル/ツールウィンドウ/アクション/設定を検索します。


def print_hi(name):
    # スクリプトをデバッグするには以下のコード行でブレークポイントを使用してください。
    print(f'Hi, {name}')  # Ctrl+F8を押すとブレークポイントを切り替えます。


# ガター内の緑色のボタンを押すとスクリプトを実行します。
if __name__ == '__main__':
    model_name = "bert-base-uncased"
    batch_size = 16
    agnews_train_loader = get_agnews_dataloader(model_name, batch_size)
    agnews_test_loader = get_agnews_dataloader(model_name, batch_size, test=True)

    batch = next(iter(agnews_train_loader))

    print(batch.keys())
    print(batch["input_ids"].shape)
    print(batch["attention_mask"].shape)
    print(batch["label"].shape)

# PyCharm のヘルプは https://www.jetbrains.com/help/pycharm/ を参照してください
