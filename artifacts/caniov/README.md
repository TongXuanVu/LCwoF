# CAN-IoV Kaggle resume checkpoint

`task_1_epoch_30.pt` is the latest checkpoint. Its metadata identifies Task 1,
epoch 30, and seen classes `[0, 1, 2]`. Resuming from it skips completed base
training, finalizes Task 1 by selecting exemplars and saving its task checkpoint,
then proceeds to Task 2.

The checkpoint contains model weights and the base-checkpoint anchor, but not
the Adam optimizer state. The optimizer is initialized again when training
resumes.

In a fresh Kaggle session, attach the `100clientiov` dataset and run this cell:

```python
!git clone --branch tongxuanvu-resume-caniov-kaggle https://github.com/TongXuanVu/LCwoF.git /kaggle/working/LCwoF
%cd /kaggle/working/LCwoF/mini_imgnet
!python build_centralized_caniov.py --fed_root /kaggle/input/datasets/tongxuanvu/100clientiov --out_root /kaggle/working/centralized_caniov
!mkdir -p /kaggle/working/logs/lcwof_caniov/checkpoints
!cp /kaggle/working/LCwoF/artifacts/caniov/task_1_epoch_30.pt /kaggle/working/logs/lcwof_caniov/checkpoints/task_1_epoch_30.pt
!python run_caniov.py --mode resume --data_root /kaggle/working/centralized_caniov --resume_path /kaggle/working/logs/lcwof_caniov/checkpoints/task_1_epoch_30.pt
```

The data builder recreates training data under `/kaggle/working`; the dataset
itself does not need to be copied into this repository. If the Kaggle dataset
is mounted at a different location, change `--fed_root` to its actual path.
