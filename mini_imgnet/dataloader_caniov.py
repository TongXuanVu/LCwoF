import os
import torch
import numpy as np
from torch.utils.data import Dataset


class CanIoVDataset(Dataset):
    """
    Dataset wrapper for tabular CAN-bus IoV data returning dict {'data': tensor, 'label': tensor}.
    """
    def __init__(self, x, y):
        if isinstance(x, np.ndarray):
            self.x = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            self.x = x.float()
        else:
            self.x = torch.tensor(x, dtype=torch.float32)

        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).long()
        elif isinstance(y, torch.Tensor):
            self.y = y.long()
        else:
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'data': self.x[idx],
            'label': self.y[idx]
        }


class CanIoVDataManager:
    """
    Manages loading of centralized CAN-IoV (100-client) tasks and global test data.

    Khac voi CICIoT23DataManager: data_root o day tro vao thu muc da duoc GOP SAN
    (xem build_centralized_caniov.py) chu khong phai thu muc federated_data goc —
    centralized_data/centralized_task_N.pt phai duoc sinh TRUOC khi chay script nay
    bang cach gop tat ca client co shard cua task do lai.
    """
    def __init__(self, data_root, use_fewshot=False, use_10shot=False, data_dir_name=None):
        self.data_root = data_root
        if data_dir_name:
            self.centralized_dir = os.path.join(data_root, data_dir_name)
        elif use_10shot:
            self.centralized_dir = os.path.join(data_root, "10shot", "centralized_data_10shot")
        elif use_fewshot:
            self.centralized_dir = os.path.join(data_root, "fewshot", "centralized_data_fewshot")
        else:
            self.centralized_dir = os.path.join(data_root, "centralized_data")
        self.global_test_file = os.path.join(data_root, "global_test_data.pt")
        if not os.path.exists(self.global_test_file):
            self.global_test_file = os.path.join(data_root, "data", "global_test_data.pt")

        if not os.path.exists(self.centralized_dir):
            raise FileNotFoundError(
                f"Centralized data directory not found: {self.centralized_dir}. "
                f"Chay build_centralized_caniov.py truoc de gop du lieu federated "
                f"thanh centralized_task_N.pt.")
        if not os.path.exists(self.global_test_file):
            raise FileNotFoundError(f"Global test file not found: {self.global_test_file} or {os.path.join(data_root, 'global_test_data.pt')}")

        print("[DataManager] Loading global test data...")
        test_dict = torch.load(self.global_test_file, map_location="cpu", weights_only=False)
        self.test_x = test_dict["x"].float()
        self.test_y = test_dict["y"].long()
        print(f"[DataManager] Loaded test set: {self.test_x.shape[0]} samples")

        # Memory for exemplars: class_idx -> {'x': tensor, 'y': tensor}
        self.exemplar_memory = {}

    def load_task_train_data(self, task_id):
        """
        Loads training data for a specific task (1-indexed: 1 to 5 cho CAN-IoV).
        Returns x, y tensors.
        """
        path = os.path.join(self.centralized_dir, f"centralized_task_{task_id}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Task data file not found: {path}")

        data = torch.load(path, map_location="cpu", weights_only=False)
        return data["x"].float(), data["y"].long()

    def get_test_dataset(self, seen_classes, max_samples_per_class=None):
        x_filtered_list = []
        y_filtered_list = []

        for c in seen_classes:
            class_mask = (self.test_y == c)
            x_c = self.test_x[class_mask]
            y_c = self.test_y[class_mask]

            if len(x_c) > 0:
                if max_samples_per_class is not None and len(x_c) > max_samples_per_class:
                    indices = np.random.choice(len(x_c), max_samples_per_class, replace=False)
                    x_filtered_list.append(x_c[indices])
                    y_filtered_list.append(y_c[indices])
                else:
                    x_filtered_list.append(x_c)
                    y_filtered_list.append(y_c)

        if len(x_filtered_list) == 0:
            return CanIoVDataset(torch.empty(0, self.test_x.shape[1]), torch.empty(0, dtype=torch.long))

        x_filtered = torch.cat(x_filtered_list, dim=0)
        y_filtered = torch.cat(y_filtered_list, dim=0)
        return CanIoVDataset(x_filtered, y_filtered)

    def select_exemplars(self, x, y, classes, m_per_class=20):
        print(f"[DataManager] Selecting {m_per_class} exemplars per class for {classes}...")
        for c in classes:
            # Chon theo CHI SO thay vi x[class_mask] (sao chep ca lop = hang chuc GB
            # o task 1 cua IoV, ~97,7 trieu mau -> het RAM). Ket qua GIONG HET:
            # mask boolean giu nguyen thu tu nen x[mask][i] == x[idx_c[i]], va
            # np.random.choice nhan cung n_samples nen tieu thu RNG y het.
            idx_c = torch.nonzero(y == c, as_tuple=True)[0]

            n_samples = idx_c.numel()
            if n_samples == 0:
                print(f"[DataManager] Warning: No samples found for class {c}!")
                continue

            n_select = max(1, int(n_samples * 0.01))
            indices = np.random.choice(n_samples, n_select, replace=False)
            sel = idx_c[torch.from_numpy(indices)]
            self.exemplar_memory[c] = {
                'x': x[sel],
                'y': y[sel]
            }
            del idx_c, sel
        print("[DataManager] Exemplar memory updated.")

    def get_calibration_dataset(self, seen_classes, current_task_x, current_task_y, current_task_classes, m_per_class=20):
        cal_x_list = []
        cal_y_list = []

        for c in seen_classes:
            if c not in current_task_classes:
                if c in self.exemplar_memory:
                    cal_x_list.append(self.exemplar_memory[c]['x'])
                    cal_y_list.append(self.exemplar_memory[c]['y'])
                else:
                    print(f"[DataManager] Warning: Class {c} not found in exemplar memory!")

        for c in current_task_classes:
            class_mask = (current_task_y == c)
            x_c = current_task_x[class_mask]
            y_c = current_task_y[class_mask]

            n_samples = x_c.shape[0]
            if n_samples > 0:
                n_select = max(1, int(n_samples * 0.01))
                indices = np.random.choice(n_samples, n_select, replace=False)
                cal_x_list.append(x_c[indices])
                cal_y_list.append(y_c[indices])

        if len(cal_x_list) == 0:
            raise ValueError("No calibration samples could be loaded!")

        cal_x = torch.cat(cal_x_list, dim=0)
        cal_y = torch.cat(cal_y_list, dim=0)

        print(f"[DataManager] Constructed balanced calibration dataset with {cal_x.shape[0]} samples.")
        return CanIoVDataset(cal_x, cal_y)
