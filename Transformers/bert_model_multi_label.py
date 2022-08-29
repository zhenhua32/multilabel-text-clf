# region 导入模块

import json
import logging
import warnings

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
train_list = json.load(
    open("../multi_label_data/econbiz/train_data.json")
)  # change the dataset folder name [ 'reuters', 'rcv1-v2', 'econbiz', 'amazon', 'dbpedia','nyt','goemotions']
# TODO: 待优化, 这段代码, 或者下面的 df 代码有坑, 这里可是有百万级别的数据, 直接载入内存就炸了
train_data = np.array(list(map(lambda x: (list(x.values())[:2]), train_list)), dtype=object)
train_labels = np.array(list(map(lambda x: list(x.values())[2], train_list)), dtype=object)
test_list = json.load(open("../multi_label_data/econbiz/test_data.json"))  # change dataset folder name
test_data = np.array(list(map(lambda x: list(x.values())[:2], test_list)), dtype=object)
test_labels = np.array(list(map(lambda x: list(x.values())[2], test_list)), dtype=object)
print("已经将数据加载到 numpy 中")

# 处理多标签
label_encoder = MultiLabelBinarizer()
label_encoder.fit(train_labels)
train_labels_enc = label_encoder.transform(train_labels)
test_labels_enc = label_encoder.transform(test_labels)
print("已经使用 sklearn 生成多标签")

# 先用 pandas 存储起来, 后续作为 torch 的 dataset
train_df = pd.DataFrame()
train_df["text"] = train_data[:, 1]
train_df["labels"] = train_labels_enc.tolist()

test_df = pd.DataFrame()
test_df["text"] = test_data[:, 1]
test_df["labels"] = test_labels_enc.tolist()

# 先拿小批量的数据试一下
# train_df = train_df.sample(n=20000)
# test_df = test_df.sample(n=5000)

print("Number of train texts ", len(train_df["text"]))
print("Number of train labels ", len(train_df["labels"]))
print("Number of test texts ", len(test_df["text"]))
print("Number of test labels ", len(test_df["labels"]))
train_df.head()
test_df

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
    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

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

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float),
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

test_params = {"batch_size": VALID_BATCH_SIZE, "shuffle": True, "num_workers": 0}
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
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


# Test Model
def validation_on_test_data(model: BERTClass, testing_loader: DataLoader):
    outputs, targets = validation(model, testing_loader)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_avg = metrics.f1_score(targets, outputs, average="samples")
    f1_score_micro = metrics.f1_score(targets, outputs, average="micro")
    f1_score_macro = metrics.f1_score(targets, outputs, average="macro")
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
validation_on_test_data(train_model, testing_loader)
