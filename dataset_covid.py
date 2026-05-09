import os
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

NUM_DAYS = 14
NUM_DYNAMIC_VARS = 10


def process_func(path: str, train: bool = True, current_id: str = "covid_causal_14d"):
    data = pd.read_csv(path)
    data.replace("?", np.nan, inplace=True)
    observed_values = data.values.astype("float32")

    load_mask_path = "./data/covid-19/covid_mask/" + current_id + ".csv"
    print(load_mask_path)
    load_mask = pd.read_csv(load_mask_path).values.astype("float32")

    if load_mask.shape != observed_values.shape:
        raise ValueError(
            f"Mask shape {load_mask.shape} does not match data shape {observed_values.shape}."
        )

    observed_values = np.nan_to_num(observed_values).astype("float32")
    load_mask = load_mask.astype("float32")

    treatment = observed_values[:, 0]
    outcomes = observed_values[:, 1:3]
    mu = observed_values[:, 3:5]
    pop_size = observed_values[:, 5]
    dynamic_flat = observed_values[:, 6:]
    expected_dynamic = NUM_DAYS * NUM_DYNAMIC_VARS
    if dynamic_flat.shape[1] != expected_dynamic:
        raise ValueError(
            f"Expected {expected_dynamic} dynamic values, got {dynamic_flat.shape[1]}."
        )
    dynamic_seq = dynamic_flat.reshape(-1, NUM_DAYS, NUM_DYNAMIC_VARS).transpose(0, 2, 1)
    pop_seq = np.repeat(pop_size[:, None], NUM_DAYS, axis=1)[:, None, :]
    x_seq = np.concatenate([pop_seq, dynamic_seq], axis=1)

    treatment_mask = load_mask[:, 0]
    outcomes_mask = load_mask[:, 1:3]
    mu_mask = load_mask[:, 3:5]
    pop_mask = load_mask[:, 5]
    dynamic_mask_flat = load_mask[:, 6:]
    dynamic_mask = dynamic_mask_flat.reshape(-1, NUM_DAYS, NUM_DYNAMIC_VARS).transpose(0, 2, 1)
    pop_seq_mask = np.repeat(pop_mask[:, None], NUM_DAYS, axis=1)[:, None, :]
    x_mask = np.concatenate([pop_seq_mask, dynamic_mask], axis=1)

    # Keep previous behavior: at validation/test, both outcomes are fully masked targets.
    if not train:
        outcomes_mask = np.zeros_like(outcomes_mask, dtype="float32")

    return {
        "treatment": treatment.astype("float32"),
        "outcomes": outcomes.astype("float32"),
        "mu": mu.astype("float32"),
        "x_seq": x_seq.astype("float32"),
        "x_mask": x_mask.astype("float32"),
        "treatment_mask": treatment_mask.astype("float32"),
        "outcomes_mask": outcomes_mask.astype("float32"),
        "mu_mask": mu_mask.astype("float32"),
    }


class tabular_dataset(Dataset):
    def __init__(
        self,
        use_index_list=None,
        missing_ratio=0.1,
        seed=0,
        train=True,
        current_id="covid_causal_14d",
    ):
        np.random.seed(seed)

        dataset_path = "./data/covid-19/covid_norm_data/" + current_id + ".csv"
        print("dataset_path", dataset_path)

        processed_data_path = (
            f"./data/covid-19/missing_ratio-{missing_ratio}_seed-{seed}_"
            f"current_id-{current_id}_train-{int(train)}.pk"
        )

        if not os.path.isfile(processed_data_path):
            self.processed = process_func(dataset_path, train=train, current_id=current_id)
            with open(processed_data_path, "wb") as f:
                pickle.dump(self.processed, f)
            print("--------Dataset created--------")
        else:
            with open(processed_data_path, "rb") as f:
                self.processed = pickle.load(f)
            print("--------Dataset loaded from cache--------")

        # Backward-compatibility for old cached format.
        if not isinstance(self.processed, dict):
            self.processed = process_func(dataset_path, train=train, current_id=current_id)
            with open(processed_data_path, "wb") as f:
                pickle.dump(self.processed, f)

        self.eval_length = NUM_DAYS

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.processed["treatment"]))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        x_seq = self.processed["x_seq"][index]
        return {
            "treatment": self.processed["treatment"][index],
            "outcomes": self.processed["outcomes"][index],
            "mu": self.processed["mu"][index],
            "x_seq": x_seq,
            "x_mask": self.processed["x_mask"][index],
            "treatment_mask": self.processed["treatment_mask"][index],
            "outcomes_mask": self.processed["outcomes_mask"][index],
            "mu_mask": self.processed["mu_mask"][index],
            "timepoints": np.arange(self.eval_length),
            "x_prop": x_seq.reshape(-1).astype("float32"),
        }

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(
    seed=1,
    batch_size=16,
    missing_ratio=0.1,
    current_id="covid_causal_14d",
):
    dataset = tabular_dataset(
        missing_ratio=missing_ratio,
        seed=seed,
        current_id=current_id,
    )
    print(f"Dataset size:{len(dataset)} entries")

    indlist = np.arange(len(dataset))
    tsi = int(len(dataset) * 0.8)
    if tsi % 8 == 1 or int(len(dataset) * 0.2) % 8 == 1:
        tsi = tsi + 3
    print("test start index", tsi)

    test_index = indlist[tsi:]
    remain_index = np.arange(0, tsi)
    np.random.shuffle(remain_index)
    train_index = remain_index[:tsi]
    valid_index = remain_index[: int(tsi * 0.1)]

    train_dataset = tabular_dataset(
        use_index_list=train_index,
        missing_ratio=missing_ratio,
        seed=seed,
        current_id=current_id,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = tabular_dataset(
        use_index_list=valid_index,
        missing_ratio=missing_ratio,
        seed=seed,
        train=False,
        current_id=current_id,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = tabular_dataset(
        use_index_list=test_index,
        missing_ratio=missing_ratio,
        seed=seed,
        train=False,
        current_id=current_id,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader
