import os
import re
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


def _list_time_dirs(root: str, prefix: str) -> List[str]:
    """
    在 root 下列出以 prefix 开头的时间目录（如 S1_YYYY_MM），并按时间升序返回目录名列表。
    """
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{4}})_(\d{{2}})$")  # e.g., S1_2020_03
    time_dirs = []
    for d in os.listdir(root):
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        m = pat.match(d)
        if m:
            year, month = int(m.group(1)), int(m.group(2))
            time_dirs.append((year, month, d))
    # 升序排序
    time_dirs.sort(key=lambda x: (x[0], x[1]))
    return [d for (_, _, d) in time_dirs]


def _pick_timesteps(sorted_keys: List[str], timestep: int) -> List[str]:
    """
    根据 timestep 从已排序的时间键（目录名）中等距抽取。
    """
    n = len(sorted_keys)
    if n == 0:
        return []
    if timestep <= 1:
        return [sorted_keys[-1]]
    if timestep >= n:
        return sorted_keys[:]  # 全部
    idxs = np.linspace(0, n - 1, num=timestep, dtype=int)
    idxs = np.unique(idxs)
    while len(idxs) < timestep:
        idxs = np.append(idxs, n - 1)
        idxs = np.unique(idxs)
    return [sorted_keys[i] for i in idxs.tolist()]


def _read_tif(path: str) -> np.ndarray:
    """
    读取 GeoTIFF 为 (C, H, W) 的 numpy array，float32；若有 NaN 用 0 替换。
    """
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
    arr = arr.astype(np.float32)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    return arr


def _read_label_tif(path: str) -> np.ndarray:
    """
    读取 label 为 (1, H, W) 的 numpy array。
    """
    with rasterio.open(path) as src:
        arr = src.read(1)  # (H, W)
    arr = arr.astype(np.float32, copy=False)
    return arr[None, ...]  # (1, H, W)


class RSDataset(Dataset):
    """
    数据集结构：
        root/
          label/{index}.tif
          HR_YYYY_MM/{index}.tif
          S1_YYYY_MM/{index}.tif ...
          train.txt / test.txt

    返回：
        HR_t1:  (C, H, W)
        S1_t2:  (T, C, H, W)
        GT:     (1, H, W)
    """
    CLASSES = [
        'un-changed', 'changed'
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        timestep: int = 1,
        hr_prefix: str = "HR",
        s1_prefix: str = "S1",
        label_dirname: str = "label",
        index_file_map: Optional[dict] = None,
        mode: str = "train",
    ):
        super().__init__()
        assert split in ("train", "test")
        self.root = root
        self.split = split
        self.timestep = int(timestep)
        self.hr_prefix = hr_prefix
        self.s1_prefix = s1_prefix
        self.index_file_map = index_file_map
        self.mode = mode

        # 读取 train/test.txt
        split_file = os.path.join(root, f"{split}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"找不到划分文件：{split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            indices = [ln.strip() for ln in f if ln.strip()]
        self.indices = [
            (self.index_file_map[i] if (self.index_file_map and i in self.index_file_map) else
             (i if i.lower().endswith(".tif") else f"{i}.tif"))
            for i in indices
        ]

        # 标签目录（绝对路径）
        self.label_dir = os.path.join(root, label_dirname)

        # HR 目录
        hr_dirs_rel = _list_time_dirs(root, self.hr_prefix)
        if len(hr_dirs_rel) == 0:
            raise RuntimeError(f"未发现 {self.hr_prefix}_YYYY_MM 目录")
        if len(hr_dirs_rel) > 1:
            print(f"[Warn] 检测到多个 {self.hr_prefix}_YYYY_MM，使用最后一个：{hr_dirs_rel[-1]}")
        self.hr_dir = os.path.join(root, hr_dirs_rel[-1])  # 绝对路径

        # S1 时间目录
        s1_dirs_rel = _list_time_dirs(root, self.s1_prefix)
        s1_pick = _pick_timesteps(s1_dirs_rel, self.timestep)
        if len(s1_pick) != self.timestep:
            print(f"[Warn] 可用时刻少于 timestep={self.timestep}，S1={len(s1_pick)}")
        self.s1_dirs = [os.path.join(root, d) for d in s1_pick]

    def __len__(self) -> int:
        return len(self.indices)

    def _normalize(self, arr: np.ndarray, scale: float = 1.0, add: float = 0.0) -> np.ndarray:
        return (arr + add) / scale if scale > 0 else arr

    def _ensure_exists(self, folder: str, filename: str) -> str:
        """folder 已经是绝对路径"""
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"缺少文件：{path}")
        return path

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index_name = self.indices[idx]

        # 标签
        label_path = self._ensure_exists(self.label_dir, index_name)
        gt = _read_label_tif(label_path)

        # HR
        hr_path = self._ensure_exists(self.hr_dir, index_name)
        hr = _read_tif(hr_path)
        hr = self._normalize(hr, scale=255.0)
        HR_t1 = torch.from_numpy(hr)

        # S1 时序
        s1_list = []
        for d in self.s1_dirs:
            s1_path = self._ensure_exists(d, index_name)
            s1 = _read_tif(s1_path)
            s1 = self._normalize(s1, scale=255.0)
            s1_list.append(s1[None, ...])
        S1_t2 = torch.from_numpy(np.concatenate(s1_list, axis=0)) if s1_list else torch.zeros((0, 0, 0, 0))

        GT = torch.from_numpy(gt).squeeze(0).long()
        if self.mode == "train":
            return HR_t1, S1_t2, GT
        else:
            return HR_t1, S1_t2, GT, index_name


# =========================
# 可视化部分
# =========================
import math
import matplotlib.pyplot as plt

def normalize_to_255(arr):
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)

def show_image(ax, arr, title=""):
    if arr.shape[0] >= 3:  # 取前三波段
        img = arr[:3].transpose(1, 2, 0)
    else:  # 单波段灰度
        img = arr[0]
    img = normalize_to_255(img)
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def visualize_sample(hr, s1, gt, timestep):
    max_cols = 4
    fig_rows = 1 + math.ceil(timestep / max_cols)
    fig_cols = max(max_cols, min(timestep, max_cols))
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 3, fig_rows * 3))
    axes = np.array(axes).reshape(fig_rows, fig_cols)

    for ax in axes.ravel():
        ax.axis("off")

    # HR + GT
    show_image(axes[0, 0], hr.numpy(), "HR_t1")
    gt_np = gt.numpy()
    if gt_np.ndim == 3:
        gt_img = gt_np[0]
    else:
        gt_img = gt_np
    axes[0, 1].imshow(gt_img, cmap="gray")
    axes[0, 1].set_title("GT", fontsize=10)
    axes[0, 1].axis("off")

    # S1 时序展示（仅展示前两个通道）
    for t in range(min(timestep, s1.shape[0])):
        r = 1 + t // fig_cols
        c = t % fig_cols
        arr = s1[t].numpy()
        if arr.shape[0] > 1:
            arr = arr[:1]  # 显示第一通道
        show_image(axes[r, c], arr, f"S1_t{t+1}")

    plt.tight_layout()
    plt.show()


# =========================
# 使用示例
# =========================
if __name__ == "__main__":
    root = r"MCD/DATA/Ukraine"
    ds_test = RSDataset(root=root, split="test", timestep=1, mode="inf")
    print("Test size:", len(ds_test))

    hr, s1, gt, index_name = ds_test[200]
    print("HR_t1:", hr.shape)
    print("S1_t2:", s1.shape)
    print("GT:", gt.shape)
    print("Index name:", index_name)

    # visualize_sample(hr, s1, gt, timestep=s1.shape[0])
