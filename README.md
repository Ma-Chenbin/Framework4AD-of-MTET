# Two-Stage Framework for Automatic Diagnosis of Multi-task in Essential Tremor via Multi-Sensory Fusion Parameters
This is  the offical code repository accompanying our paper on **Two-Stage Framework for Automatic Diagnosis of Multi-task in Essential Tremor via Multi-Sensory Fusion Parameters**.

For a detailed description of technical details and experimental results, please refer to our paper:

Chenbin Ma, [Two-Stage Framework for Automatic Diagnosis of Multi-task in Essential Tremor via Multi-Sensory Fusion Parameters](https://doi.org/******), ****** **!!!** (2022) ******.
    
    @article{Chenbin Ma:2021Self,
        doi = {=======},
        url = {=======},
        year = {2022},
        month = =======,
        publisher = {=======},
        volume = {=======},
        pages = {=======},
        author = {Chenbin Ma, Peng Zhang, Longsheng Pan, Xuemei Li, Chunyu Yin, Ailing Li, Rui Zong, Zhengbo Zhang},
        title = {Two-Stage Framework for Automatic Diagnosis of Multi-task in Essential Tremor via Multi-Sensory Fusion Parameters},
        journal = {=======},
        eprint = {=======},
        archivePrefix={=======},
        primaryClass={=======}
    }

## We will continue to update the experiment
To our knowledge, this study built the first multitasking assessment database for patients with essential tremors, which included 121 subjects who were screened in hospitals and diagnosed by neurologist committees, and multi-modal data from laboratory screening were recorded throughout. **Our dataset is the most extensive compared to state-of-the-art works**. In addition, due to stringent restrictions on patient privacy, ethical approval, and hospital licensing, **there is no corresponding publicly available datasets for additional training or validation**. It is important to note that there are many difficulties in collecting data, and despite trying to collect data continuously to obtain a larger dataset, few patients chose to be seen during this period due to the impact of the Coronavirus (COVID-19).

## Usage information
### Preparation
1. install dependencies from `TSF4ET_cls.yml` by running `conda env create -f TSF4ET_cls.yml` and activate the environment via `conda activate TSF4ET_cls`
2. follow the instructions in `data_preprocessing.ipynb` on how to download and preprocess the ET_tremor datasets; in the following, we assume for definiteness that the preprocessed dataset folders can be found at `./data/rest`,`./data/arm`,`./data/fingerfinger`,`./data/wing` and `./data/fingernose`.

### 1. Pretraining PTRS on All
Pretrain a PTRS model on the full dataset collection (will take about 2 days on a Tesla V100 GPU):
`python main_PTRS_lightning.py --data ./data/rest --data ./data/arm --data ./data/fingerfinger --data ./data/wing --data ./data/fingernose --normalize --epochs 1000 --output-path=./runs/PTRS/all --lr 0.0001 --batch-size 32 --input-size 1000 --fc-encoder --negatives-from-same-seq-only`

### 2. Finetuning QSQ on Specific dataset
1. Finetune just the classification head:
`matlab SQSmodeling.mlx`
2. Finetune the entire model with discriminative learning rates:
`matlab SQSmodeling.mlx`

The script prints the results and saves the finetuned model in a new folder next to the pretrained model and a pickle file containing the results of the evaluation.



## Pretrained models
For each method (SVC, Ensembles, Tree, etc.), we provide the best-performing pretrained model after pretraining on *All*: [link to datacloud](https://github.com/Ma-Chenbin/Two-Stage-Framework-for-AD-of-Multi-task-in-ET-via-Multi-Sensory-Fusion-Parameters/data). More models are available from the authors upon request.

