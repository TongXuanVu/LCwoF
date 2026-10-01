"""Gop du lieu federated 100-client CAN-IoV thanh du lieu TAP TRUNG (centralized)
cho LCwoF.

Bo IoV goc la federated: moi client co shard rieng
    federated_data/client_{c}_task_{t}.pt            (day du)
    federated_data_fewshot/client_{c}_task_{t}.pt     (1%, CHI task 2..5)
    federated_data_10shot/client_{c}_task_{t}.pt      (10-shot, CHI task 2..5)
LCwoF (ban goc) la bai TAP TRUNG (khong lien ket) nen can GOP tat ca client lai
thanh 1 file duy nhat moi task: centralized_task_{t}.pt = {"x":..., "y":...}.

Quy uoc giong het AFSIC-IOV/tools/make_fewshot_iov100.py va readme LCwoF (CIC-IoT23):
  - Task 1 la task base, CA BA kich ban (full / 1% / 10shot) deu dung chung du
    lieu FULL cua task 1 (fewshot/10shot KHONG sinh rieng task 1).
  - Client "gia nhap dan theo task" (xem CLAUDE.md): task t chi gop nhung client
    THAT SU co file client_*_task_t.pt, khong ep ve 100 client.

Chay TRONG notebook Kaggle (truoc khi goi run_caniov.py), vi sandbox khong ghi
duoc vao noi chua du lieu goc:

    python build_centralized_caniov.py \
        --fed_root /kaggle/input/datasets/tongxuanvu/100clientiov \
        --out_root /kaggle/working/centralized_caniov

Ket qua:
    out_root/global_test_data.pt                              (copy nguyen ban)
    out_root/centralized_data/centralized_task_{1..5}.pt       (full)
    out_root/fewshot/centralized_data_fewshot/centralized_task_{1..5}.pt
    out_root/10shot/centralized_data_10shot/centralized_task_{1..5}.pt
"""
import argparse
import glob
import os
import re
import shutil
import time

import torch

NUM_TASKS = 5
TASK_BASE = 1


def gop_task(files, task_id):
    """Doc tat ca shard cua 1 task, noi x/y lai thanh 1 tensor duy nhat."""
    xs, ys = [], []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        xs.append(d["x"])
        ys.append(d["y"])
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return {"x": x, "y": y}


def xu_ly_kich_ban(src_dir, dst_dir, tasks, dry_run=False):
    """Gop 1 kich ban (full/fewshot/10shot) cho danh sach task da cho."""
    if not dry_run:
        os.makedirs(dst_dir, exist_ok=True)
    for t in tasks:
        pattern = os.path.join(src_dir, f"client_*_task_{t}.pt")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  [CANH BAO] khong tim thay shard nao cho task {t} o {src_dir}")
            continue
        t0 = time.time()
        merged = gop_task(files, t)
        n_clients = len(files)
        n_samples = merged["y"].shape[0]
        classes = sorted(set(merged["y"].tolist()))
        print(f"  task {t}: {n_clients} client, {n_samples} mau, lop {classes} "
              f"({time.time()-t0:.1f}s)")
        if not dry_run:
            out_path = os.path.join(dst_dir, f"centralized_task_{t}.pt")
            torch.save(merged, out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fed_root", required=True,
                     help="Thu muc goc dataset 100clientiov (chua federated_data/, "
                          "federated_data_fewshot/, federated_data_10shot/, global_test_data.pt)")
    ap.add_argument("--out_root", required=True,
                     help="Thu muc dich de ghi du lieu da gop (vd /kaggle/working/centralized_caniov)")
    ap.add_argument("--dry-run", action="store_true", help="Chi bao cao, khong ghi file")
    args = ap.parse_args()

    fed_root = args.fed_root
    out_root = args.out_root

    global_test = os.path.join(fed_root, "global_test_data.pt")
    if not os.path.exists(global_test):
        raise SystemExit(f"Khong thay global_test_data.pt o {global_test}")

    if not args.dry_run:
        os.makedirs(out_root, exist_ok=True)
        shutil.copy(global_test, os.path.join(out_root, "global_test_data.pt"))
        print(f"Da copy global_test_data.pt -> {out_root}")

    all_tasks = list(range(1, NUM_TASKS + 1))
    fs_tasks = [t for t in all_tasks if t != TASK_BASE]

    print("\n=== Kich ban FULL (task 1..5) ===")
    xu_ly_kich_ban(os.path.join(fed_root, "federated_data"),
                    os.path.join(out_root, "centralized_data"),
                    all_tasks, args.dry_run)

    print("\n=== Kich ban FEWSHOT 1% (task 2..5; task 1 dung chung ban full) ===")
    dst_fewshot = os.path.join(out_root, "fewshot", "centralized_data_fewshot")
    if not args.dry_run:
        os.makedirs(dst_fewshot, exist_ok=True)
        shutil.copy(
            os.path.join(out_root, "centralized_data", f"centralized_task_{TASK_BASE}.pt"),
            os.path.join(dst_fewshot, f"centralized_task_{TASK_BASE}.pt"))
        print(f"  task {TASK_BASE}: dung chung ban full (copy)")
    xu_ly_kich_ban(os.path.join(fed_root, "federated_data_fewshot"),
                    dst_fewshot, fs_tasks, args.dry_run)

    print("\n=== Kich ban 10SHOT (task 2..5; task 1 dung chung ban full) ===")
    dst_10shot = os.path.join(out_root, "10shot", "centralized_data_10shot")
    if not args.dry_run:
        os.makedirs(dst_10shot, exist_ok=True)
        shutil.copy(
            os.path.join(out_root, "centralized_data", f"centralized_task_{TASK_BASE}.pt"),
            os.path.join(dst_10shot, f"centralized_task_{TASK_BASE}.pt"))
        print(f"  task {TASK_BASE}: dung chung ban full (copy)")
    xu_ly_kich_ban(os.path.join(fed_root, "federated_data_10shot"),
                    dst_10shot, fs_tasks, args.dry_run)

    print(f"\nXong. Du lieu da gop nam o: {out_root}")


if __name__ == "__main__":
    main()
