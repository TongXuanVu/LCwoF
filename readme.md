## Generalized and Incremental Few-Shot Learning by Explicit Learning and Calibration without Forgetting
Official Implementation, ICCV21

[arxiv](https://arxiv.org/abs/2108.08165)  
[Computer Vision Talks](https://www.youtube.com/watch?v=i6ZbnnKIACI)  
[poster](https://drive.google.com/file/d/1AaVD1x22c3wi0tNjmwtP1MLBpy8PnNin/view?usp=sharing)  |  [poster_slides](https://drive.google.com/file/d/18rVouHgWbUT5voy-vN4MdC_sRuMCzDTZ/view?usp=sharing) | [5_min_video](https://drive.google.com/file/d/1oFjWyuCM60XHfPbAKNLU7JwzBcSNHOVO/view?usp=sharing)


##### Data
[miniImageNet](https://drive.google.com/file/d/1CZPTOfQMp5ANF-BIuK9O9NdcPlT5XMHE/view?usp=sharing)  
here you can find all the splits and files for episodic training on mini-ImageNet (2.1 Gb)  

[pretrained_model](https://drive.google.com/file/d/165yPQtX1pWPZR_rBdPih2Rl1Xq6G7ln3/view?usp=sharing) on base 64 classes  
there is also script to train the model from scratch  
`python mini_imgnet/run_pretrain_base.py`

#### Run
set up all paths and run

`python mini_imagenet/run_novel.py
`


##### Logs

set up params for logging of training, metrics and visdom in `run_novel.py`


[files with harmonic mean and arithmetic mean](https://drive.google.com/file/d/1TIjWIOXzxPcHAfa1VTTk3p1OvcbvsNEK/view?usp=sharing) in different spaces (generalized and not), for 5w1s   
to save these files yourself turn `write_in_file=True`


###### details

pytorch 1.6  
[list of all packages that were installed](https://drive.google.com/file/d/178AdC8oQNJtJMeR78Ay4mdqhfJYWQyuY/view?usp=sharing) but you do not need all of them [just in case]

#### cite

@inproceedings{kukleva2021lcwof,  
    author    = {Kukleva, Anna and Kuehne, Hilde and Schiele, Bernt},  
    title     = {Generalized and Incremental Few-Shot Learning by Explicit Learning and Calibration Without Forgetting},  
    booktitle = {ICCV},  
    year      = {2021},  
}

---

## 🚀 Running on Kaggle (CIC-IoT23 Centralized Split)

Follow these steps to run the training, validation, and resume modes on Kaggle with the dataset **`tongxuanvu/dataset-fl`**.

### 1. Setup Notebook
Create a new Python notebook on Kaggle. In the notebook settings (right sidebar):
* Enable **GPU T4 x2** or **GPU P100** under Accelerator.
* Add the dataset: `tongxuanvu/dataset-fl`.

### 2. Clone Repository
Run the following shell commands in a notebook cell:
```bash
!git clone https://github.com/TongXuanVu/LCwoF.git
%cd LCwoF/mini_imgnet
```

### 3. Run Commands

#### **A. Train Mode (Full 330 Rounds)**
```bash
!python run_ciciot23.py \
    --mode train \
    --data_root /kaggle/input/datasets/tongxuanvu/dataset-fl \
    --run_dir /kaggle/working/logs/lcwof_ciciot23
```

#### **B. Debug Mode (Fast test, 2 epochs per phase, 200 samples/class test set)**
```bash
!python run_ciciot23.py \
    --mode train \
    --debug \
    --data_root /kaggle/input/datasets/tongxuanvu/dataset-fl \
    --run_dir /kaggle/working/logs/lcwof_ciciot23
```

#### **C. Test Mode (Evaluate all checkpoints)**
```bash
!python run_ciciot23.py \
    --mode test \
    --data_root /kaggle/input/datasets/tongxuanvu/dataset-fl \
    --test_dir /kaggle/working/logs/lcwof_ciciot23/seed42_<timestamp>/
```

#### **D. Resume Mode (Resume from a specific checkpoint)**
```bash
!python run_ciciot23.py \
    --mode resume \
    --data_root /kaggle/input/datasets/tongxuanvu/dataset-fl \
    --resume_path /kaggle/working/logs/lcwof_ciciot23/seed42_<timestamp>/checkpoints/task_3_phase2_epoch_10.pt
```

### 📈 Metrics and Plots Output
All metrics are stored in `/kaggle/working/logs/lcwof_ciciot23/seed42_<timestamp>/round_metrics.csv` and confusion matrix heatmaps under `/kaggle/working/logs/lcwof_ciciot23/seed42_<timestamp>/confusion_matrices/cm_round_{round_idx}.png`.
You can download them by running:
```bash
!zip -r lcwof_results.zip /kaggle/working/logs/lcwof_ciciot23
```

---

## 🚀 Running on Kaggle (CAN-IoV 100-client Centralized Split)

Bản CAN-IoV dùng dataset gốc là FEDERATED (`tongxuanvu/100clientiov`, moi client mot
shard rieng theo task), nen phai GOP lai thanh tap trung bang
`build_centralized_caniov.py` NGAY TRONG notebook truoc khi train — khong co san
`centralized_data/` nhu ban CIC-IoT23.

5 task, 13 lop, tang dan **3,3,3,2,2** (khac CIC-IoT23 la 6 task/34 lop/6,6,6,6,5,5).
Task 1 luon dung FULL data, ca 3 kich ban (full/1%/10-shot) deu dung chung.

### 1. Setup Notebook
* Enable **GPU T4 x2** hoac **GPU P100**.
* Add dataset: `tongxuanvu/100clientiov`.

### 2. Clone Repository
```bash
!git clone https://github.com/TongXuanVu/LCwoF.git
%cd LCwoF/mini_imgnet
```

### 3. Gop du lieu federated -> centralized (bat buoc, chay 1 lan moi phien)
```bash
!python build_centralized_caniov.py \
    --fed_root /kaggle/input/datasets/tongxuanvu/100clientiov \
    --out_root /kaggle/working/centralized_caniov
```

### 4. Chay train (full / fewshot 1% / 10-shot)
```bash
# full
!python run_caniov.py --mode train \
    --data_root /kaggle/working/centralized_caniov \
    --run_dir /kaggle/working/logs/lcwof_caniov

# fewshot 1%
!python run_caniov.py --mode train --use_fewshot \
    --data_root /kaggle/working/centralized_caniov \
    --run_dir /kaggle/working/logs/lcwof_caniov_fewshot

# 10-shot
!python run_caniov.py --mode train --use_10shot \
    --data_root /kaggle/working/centralized_caniov \
    --run_dir /kaggle/working/logs/lcwof_caniov_10shot
```
Them `--debug` de chay thu nhanh (2 epoch/phase) truoc khi chay that.

### 5. Resume / Test mode
Giong het ban CIC-IoT23, chi doi `run_caniov.py` va `--data_root
/kaggle/working/centralized_caniov`:
```bash
!python run_caniov.py --mode resume \
    --data_root /kaggle/working/centralized_caniov \
    --resume_path /kaggle/working/logs/lcwof_caniov/checkpoints/task_3_phase2_epoch_10.pt

!python run_caniov.py --mode test \
    --data_root /kaggle/working/centralized_caniov \
    --test_dir /kaggle/working/logs/lcwof_caniov
```
