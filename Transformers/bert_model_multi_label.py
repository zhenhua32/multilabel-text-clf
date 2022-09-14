"""
主要验证下怎么减少内存的使用, Econbiz 这个数据集有100w的数据, 直接全部加载的话, 内存会炸
"""

# region 导入模块

import json
import logging
import warnings
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from torch import cuda
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer

import wandb

logging.basicConfig(level=logging.ERROR)
warnings.simplefilter("ignore")

# endregion


# 初始化 wandb
# wandb.init(project="multilabel", mode="disabled")
wandb.init(project="multilabel")

# region 导入数据集
dataset = "Econbiz"  # [ 'R21578', 'RCV1-V2', 'Econbiz', 'Amazon-531', 'DBPedia-298','NYT AC','GoEmotions']
labels = 5658  # [90,101,5658,512,298,166,28]
# change the dataset folder name [ 'reuters', 'rcv1-v2', 'econbiz', 'amazon', 'dbpedia','nyt','goemotions']
train_file = "../multi_label_data/econbiz/train_data.json"
test_file = "../multi_label_data/econbiz/test_data.json"

# 初始化 MultiLabelBinarizer, 用训练数据 fit 下.
# 本身数据量变成 pandas 是不太的, 大的是 MultiLabelBinarizer 处理后的, 因为类多, 所以生成的矩阵非常大
train_list = json.load(open(train_file, "r", encoding="utf-8"))
train_df = pd.DataFrame(train_list)
test_list = json.load(open(test_file, "r", encoding="utf-8"))
test_df = pd.DataFrame(test_list)
label_encoder = MultiLabelBinarizer()
label_encoder.fit(train_df["labels"])
# endregion

#  Setting up the device for GPU usage
device = "cuda" if cuda.is_available() else "cpu"

# 定义超参数
# Defining some key variables that will be used later on in the training
MAX_LEN = 64
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 256
EPOCHS = 15  # epochs
LEARNING_RATE = 1e-05
HF_MODEL = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(HF_MODEL, do_lower_case=True, padding=True)
wandb.config.update(
    {
        "max_len": MAX_LEN,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "valid_batch_size": VALID_BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
    }
)


# region 将数据集转换成 torch 的 Dataset 类, 并使用 DataLoader
class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df = df
        self.text = self.df["text"]
        self.labels = self.df["labels"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # 每次处理一条数据
        targets = label_encoder.transform([self.labels[index]])[0]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.float),
        }


# Creating the dataset and dataloader for the neural network
# Train-Val-Test Split
train_size = 0.8
train_dataset = train_df.sample(frac=train_size, random_state=200)
valid_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)
test_dataset = test_df.reset_index(drop=True)

print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VAL Dataset: {}".format(valid_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
validation_set = CustomDataset(valid_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

# Load Data
train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}
test_params = {"batch_size": VALID_BATCH_SIZE, "shuffle": False, "num_workers": 0}
training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **test_params)
testing_loader = DataLoader(testing_set, **test_params)

# endregion


# region 定义模型
# 模型本身还是非常简单的结构
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(HF_MODEL)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, labels)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        output_2 = self.l2(pooler)
        output = self.l3(output_2)
        return output


model = BERTClass()
model.to(device)
wandb.watch(model)
# endregion


# Define Loss function
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# Plot Val loss
def loss_plot(epochs, loss):
    plt.plot(epochs, loss, color="red", label="loss")
    plt.xlabel("epochs")
    plt.title("validation loss")
    plt.savefig(dataset + "_val_loss.png")


def validation(model: BERTClass, testing_loader: DataLoader):
    """
    执行推理, 获取模型的输出和实际标签
    """
    model.eval()
    # 注意需要将 shape 修改为 (0, 标签数), 方便后面使用 concatenate
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0), total=len(testing_loader)):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            # 这个操作太慢了, 拖慢了速度
            fin_targets.append(targets.cpu().detach().numpy())
            fin_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())
    fin_targets = np.concatenate(fin_targets)
    fin_outputs = np.concatenate(fin_outputs)
    return fin_outputs, fin_targets


# Test Model
def validation_on_test_data(model: BERTClass, testing_loader: DataLoader):
    # TODO: 生成的矩阵可能太大了, 无法一次性加载进内存
    outputs, targets = validation(model, testing_loader)
    print("已经完成推理")
    outputs = np.array(outputs) >= 0.5
    # TODO sklearn 也会造成内存爆炸, 而且很慢
    start = time.time()
    accuracy = metrics.accuracy_score(targets, outputs)
    print(time.time() - start)
    start = time.time()
    f1_score_avg = metrics.f1_score(targets, outputs, average="samples")
    print(time.time() - start)
    start = time.time()
    f1_score_micro = metrics.f1_score(targets, outputs, average="micro")
    print(time.time() - start)
    start = time.time()
    f1_score_macro = metrics.f1_score(targets, outputs, average="macro")
    print(time.time() - start)
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Samples) = {f1_score_avg}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    wandb.log(
        {
            "accuracy": accuracy,
            "f1_score_avg": f1_score_avg,
            "f1_score_micro": f1_score_micro,
            "f1_score_macro": f1_score_macro,
        }
    )

    # Save results
    with open(dataset + "_results.txt", "w") as f:
        print(
            f"F1 Score (Samples) = {f1_score_avg}",
            f"Accuracy Score = {accuracy}",
            f"F1 Score (Micro) = {f1_score_micro}",
            f"F1 Score (Macro) = {f1_score_macro}",
            file=f,
        )


# Train Model
def train_model(
    start_epochs: int,
    n_epochs: int,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    model: BERTClass,
    optimizer: torch.optim.Optimizer,
):
    loss_vals = []  # 验证集的损失
    for epoch in range(start_epochs, n_epochs + 1):
        train_loss = 0
        valid_loss = 0
        total_train = 0
        total_valid = 0

        ######################
        # Train the model #
        ######################

        model.train()
        print("############# Epoch {}: Training Start   #############".format(epoch))
        for batch_idx, data in tqdm(enumerate(training_loader), total=len(training_loader)):
            optimizer.zero_grad()

            # Forward
            ids = data["ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            # Backward
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            # 应该怎么计算 loss, 让训练集的 loss 和验证集的 loss 可以比较
            # 当前这个公式就是前面的 train_loss 的和除以 batch_idx
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            total_train += loss.item()

            if batch_idx % 100 == 0:
                wandb.log({"train_loss": train_loss, "batch_idx": batch_idx, "epoch": epoch})

        ######################
        # Validate the model #
        ######################

        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data["ids"].to(device, dtype=torch.long)
                targets = data["targets"].to(device, dtype=torch.float)
                mask = data["mask"].to(device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                total_valid += loss.item()

                if batch_idx % 100 == 0:
                    wandb.log({"valid_loss": valid_loss, "batch_idx": batch_idx, "epoch": epoch})

            # calculate average losses
            print(train_loss, len(training_loader), total_train, total_train / len(training_loader))
            print(valid_loss, len(validation_loader), total_valid, total_valid / len(validation_loader))
            train_loss = train_loss / len(training_loader)
            valid_loss = valid_loss / len(validation_loader)
            # print training/validation statistics
            print(
                "Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}".format(
                    epoch, train_loss, valid_loss
                )
            )
            loss_vals.append(valid_loss)

            wandb.log({"train_loss": train_loss, "valid_loss": valid_loss, "epoch": epoch})

        # 每次执行完后评估一下
        validation_on_test_data(model, validation_loader)

    # Plot loss
    loss_plot(np.linspace(1, n_epochs, n_epochs).astype(int), loss_vals)
    return model


# 训练模型
trained_model = train_model(1, EPOCHS, training_loader, validation_loader, model, optimizer)
validation_on_test_data(trained_model, testing_loader)
