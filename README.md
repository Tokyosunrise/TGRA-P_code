# TGRA-P: Task-driven Model Predicts 90-Day Mortality from ICU Clinical Notes on Mechanical Ventilation

This repo hosts pretraining and finetuning scripts for TGRA-P.

## Requirements

```
torch
argparse
copy
tqdm
matplotlib
numpy
pandas
time
sklearn
inspect
Ranger
```

## Pretraining the model

We provide a notebook (pretrain-xlnet.ipynb) to pretrain your own Clinical model.

## Using Finetuned weights for Mortality Prediction
```
python train.py \
  --data_dir DATA_FILE\
  --config_path CONFIG\
  --model_path MORTALITY/PMV_MODEL_PATH \
  --save_meta_finetune_path SAVE_PATH \
  --prediction_label Mortality\
  --Batch_Size_Meta 4 \
  --Learning_Rate_Meta 1e-5 \
  --Training_Epoch_Meta 4 \
  --Batch_Size_Finetune 64 \
  --Learning_Rate_Finetune 2e-5 \
  --Training_Epoch_Finetune 30 \
  --saving_notes_embed_batch_size 16 \
  --skip_meta_finetuned 
```

## Datasets

We use [MIMIC-III](https://mimic.physionet.org/about/mimic/). 

```
-data
   -train.csv
   -val.csv
   -test.csv
```



