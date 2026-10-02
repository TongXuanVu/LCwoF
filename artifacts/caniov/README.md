# CAN-IoV Kaggle resume checkpoint

`task_1_epoch_29.pt` is the checkpoint produced by the Kaggle run that timed out
after 42,358 seconds, at the end of Task 1 Epoch 29. Its metadata identifies
Task 1, epoch 29, and seen classes `[0, 1, 2]`. The repository's resume code
continues at Epoch 30 and then proceeds to later tasks.

The checkpoint contains model weights and the base-checkpoint anchor, but does
not contain the Adam optimizer state. The optimizer is therefore initialized
again when training resumes.

To resume on Kaggle, clone this branch, rebuild the centralized data from the
attached `100clientiov` dataset, copy the checkpoint into the run's checkpoint
directory, and invoke `run_caniov.py` from `mini_imgnet`:

```python
!git clone --branch tongxuanvu-resume-caniov-kaggle https://github.com/TongXuanVu/LCwoF.git
%cd /kaggle/working/LCwoF/mini_imgnet
!python build_centralized_caniov.py --fed_root /kaggle/input/datasets/tongxuanvu/100clientiov --out_root /kaggle/working/centralized_caniov
!mkdir -p /kaggle/working/logs/lcwof_caniov/checkpoints
!cp /kaggle/working/LCwoF/artifacts/caniov/task_1_epoch_29.pt /kaggle/working/logs/lcwof_caniov/checkpoints/task_1_epoch_29.pt
!python run_caniov.py --mode resume --data_root /kaggle/working/centralized_caniov --resume_path /kaggle/working/logs/lcwof_caniov/checkpoints/task_1_epoch_29.pt
```

The data builder recreates training data under `/kaggle/working`; the dataset
itself does not need to be copied into this repository.
