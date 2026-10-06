# Checkpoint nen task 1 (CAN-IoV) cho kich ban 1% va 10shot

`checkpoint_task_1.pt.gz` = `checkpoint_task_1.pt` cua lan chay FULL (xong task 1, 30 epoch, acc 83.45%),
nen gzip (129MB -> 33MB vi GitHub gioi han 100MB/file).
sha256 cua file sau khi giai nen: b0cc7188def595c1191e1bc4a1965c7c4eb276517137d7a5f4c0821f97ba69dd

Task 1 la task base, CA BA kich ban (full / 1% / 10shot) dung chung du lieu FULL cua task 1
(xem build_centralized_caniov.py), nen 1% va 10shot resume tu day roi chi chay task 2..5.

Giai nen:  gzip -dc checkpoint_task_1.pt.gz > checkpoint_task_1.pt
Chay:      python run_caniov.py --mode resume --resume_path <run_dir>/checkpoint_task_1.pt --use_fewshot|--use_10shot ...
Gop du lieu (khong gop ban full ~18GB):  python build_centralized_caniov.py --fed_root ... --out_root ... --skip_full
