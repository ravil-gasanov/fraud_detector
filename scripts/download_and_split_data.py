import shutil

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split


def download_data(dataset_path: str = "data/creditcard.csv"):
    """
    Downloads the credit card fraud detection dataset from Kaggle.
    """
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    shutil.copy(path + "/creditcard.csv", dataset_path)

    print(f"Dataset downloaded and moved to {dataset_path}")


def split_train_test(dataset_path: str = "data/creditcard.csv", test_size=0.2, random_state=42):
    """
    Splits the dataset into train and test sets.
    """
    data = pd.read_csv(dataset_path)
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    dataset_dir = dataset_path.rsplit("/", 1)[0]
    train_path = f"{dataset_dir}/train.csv"
    test_path = f"{dataset_dir}/test.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print("Dataset split into train and test sets.")


if __name__ == "__main__":
    download_data()
    split_train_test()
