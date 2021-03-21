<h1 align="center">ParsNER</h1>

<br/><br/>

- [Introduction](#introduction)
- [Dataset Information](#dataset-information)
- [Evaluation](#evaluation)
  - [Bert](#bert)
  - [Roberta](#roberta)
  - [Distilbert](#distilbert)
  - [Albert](#albert)
- [How To Use](#how-to-use)
  - [Installing requirements](#installing-requirements)
  - [How to predict using pipeline](#how-to-predict-using-pipeline)
- [Models](#models)
  - [Hugging Face Model Hub](#hugging-face-model-hub)
  - [Training](#training)
- [Cite](#cite)
- [Questions?](#questions)


## Introduction

This repo contains all existing pretrained models that are fine-tuned for the Named Entity Recognition (NER) task. These models trained on a mixed NER dataset collected from [ARMAN](https://github.com/HaniehP/PersianNER), [PEYMA](http://nsurl.org/2019-2/tasks/task-7-named-entity-recognition-ner-for-farsi/), and [WikiANN](https://elisa-ie.github.io/wikiann/) that covered ten types of entities: 

- Date (DAT)
- Event (EVE)
- Facility (FAC)
- Location (LOC)
- Money (MON)
- Organization (ORG)
- Percent (PCT)
- Person (PER)
- Product (PRO)
- Time (TIM)


## Dataset Information

|       |   Records |   B-DAT |   B-EVE |   B-FAC |   B-LOC |   B-MON |   B-ORG |   B-PCT |   B-PER |   B-PRO |   B-TIM |   I-DAT |   I-EVE |   I-FAC |   I-LOC |   I-MON |   I-ORG |   I-PCT |   I-PER |   I-PRO |   I-TIM |
|:------|----------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| Train |     29133 |    1423 |    1487 |    1400 |   13919 |     417 |   15926 |     355 |   12347 |    1855 |     150 |    1947 |    5018 |    2421 |    4118 |    1059 |   19579 |     573 |    7699 |    1914 |     332 |
| Valid |      5142 |     267 |     253 |     250 |    2362 |     100 |    2651 |      64 |    2173 |     317 |      19 |     373 |     799 |     387 |     717 |     270 |    3260 |     101 |    1382 |     303 |      35 |
| Test  |      6049 |     407 |     256 |     248 |    2886 |      98 |    3216 |      94 |    2646 |     318 |      43 |     568 |     888 |     408 |     858 |     263 |    3967 |     141 |    1707 |     296 |      78 |


**Download You can download the dataset [from here](https://drive.google.com/uc?id=1fC2WGlpqumUTaT9Dr_U1jO2no3YMKFJ4)**


## Evaluation

The following tables summarize the scores obtained by pretrained models overall and per each class.

|    Model   | accuracy | precision |  recall  |    f1    |
|:----------:|:--------:|:---------:|:--------:|:--------:|
|    Bert    | 0.995086 |  0.953454 | 0.961113 | 0.957268 |
|   Roberta  | 0.994849 |  0.949816 | 0.960235 | 0.954997 |
| Distilbert | 0.994534 |  0.946326 |  0.95504 | 0.950663 |
|   Albert   | 0.993405 |  0.938907 | 0.943966 | 0.941429 |


### Bert

|     	| number 	| precision 	|  recall  	|    f1    	|
|:---:	|:------:	|:---------:	|:--------:	|:--------:	|
| DAT 	|   407  	|  0.860636 	| 0.864865 	| 0.862745 	|
| EVE 	|   256  	|  0.969582 	| 0.996094 	| 0.982659 	|
| FAC 	|   248  	|  0.976190 	| 0.991935 	| 0.984000 	|
| LOC 	|  2884  	|  0.970232 	| 0.971914 	| 0.971072 	|
| MON 	|   98   	|  0.905263 	| 0.877551 	| 0.891192 	|
| ORG 	|  3216  	|  0.939125 	| 0.954602 	| 0.946800 	|
| PCT 	|   94   	|  1.000000 	| 0.968085 	| 0.983784 	|
| PER 	|  2645  	|  0.965244 	| 0.965974 	| 0.965608 	|
| PRO 	|   318  	|  0.981481 	| 1.000000 	| 0.990654 	|
| TIM 	|   43   	|  0.692308 	| 0.837209 	| 0.757895 	|

### Roberta

|     	| number 	| precision 	|  recall  	|    f1    	|
|:---:	|:------:	|:---------:	|:--------:	|:--------:	|
| DAT 	|   407  	|  0.844869 	| 0.869779 	| 0.857143 	|
| EVE 	|   256  	|  0.948148 	| 1.000000 	| 0.973384 	|
| FAC 	|   248  	|  0.957529 	| 1.000000 	| 0.978304 	|
| LOC 	|  2884  	|  0.965422 	| 0.968100 	| 0.966759 	|
| MON 	|   98   	|  0.937500 	| 0.918367 	| 0.927835 	|
| ORG 	|  3216  	|  0.943662 	| 0.958333 	| 0.950941 	|
| PCT 	|   94   	|  1.000000 	| 0.968085 	| 0.983784 	|
| PER 	|  2646  	|  0.957030 	| 0.959562 	| 0.958294 	|
| PRO 	|   318  	|  0.963636 	| 1.000000 	| 0.981481 	|
| TIM 	|   43   	|  0.739130 	| 0.790698 	| 0.764045 	|


### Distilbert

|     	| number 	| precision 	|  recall  	|    f1    	|
|:---:	|:------:	|:---------:	|:--------:	|:--------:	|
| DAT 	|   407  	|  0.812048 	| 0.828010 	| 0.819951 	|
| EVE 	|   256  	|  0.955056 	| 0.996094 	| 0.975143 	|
| FAC 	|   248  	|  0.972549 	| 1.000000 	| 0.986083 	|
| LOC 	|  2884  	|  0.968403 	| 0.967060 	| 0.967731 	|
| MON 	|   98   	|  0.925532 	| 0.887755 	| 0.906250 	|
| ORG 	|  3216  	|  0.932095 	| 0.951803 	| 0.941846 	|
| PCT 	|   94   	|  0.936842 	| 0.946809 	| 0.941799 	|
| PER 	|  2645  	|  0.959818 	| 0.957278 	| 0.958546 	|
| PRO 	|   318  	|  0.963526 	| 0.996855 	| 0.979907 	|
| TIM 	|   43   	|  0.760870 	| 0.813953 	| 0.786517 	|

### Albert

|     	| number 	| precision 	|  recall  	|    f1    	|
|:---:	|:------:	|:---------:	|:--------:	|:--------:	|
| DAT 	|   407  	|  0.820639 	| 0.820639 	| 0.820639 	|
| EVE 	|   256  	|  0.936803 	| 0.984375 	| 0.960000 	|
| FAC 	|   248  	|  0.925373 	| 1.000000 	| 0.961240 	|
| LOC 	|  2884  	|  0.960818 	| 0.960818 	| 0.960818 	|
| MON 	|   98   	|  0.913978 	| 0.867347 	| 0.890052 	|
| ORG 	|  3216  	|  0.920892 	| 0.937500 	| 0.929122 	|
| PCT 	|   94   	|  0.946809 	| 0.946809 	| 0.946809 	|
| PER 	|  2644  	|  0.960000 	| 0.944024 	| 0.951945 	|
| PRO 	|   318  	|  0.942943 	| 0.987421 	| 0.964670 	|
| TIM 	|   43   	|  0.780488 	| 0.744186 	| 0.761905 	|

## How To Use
You use this model with Transformers pipeline for NER.

### Installing requirements

```bash
pip install sentencepiece
pip install transformers
```

### How to predict using pipeline

```python
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification  # for pytorch
from transformers import TFAutoModelForTokenClassification  # for tensorflow
from transformers import pipeline

# model_name_or_path = "HooshvareLab/bert-fa-zwnj-base-ner"  # Roberta
# model_name_or_path = "HooshvareLab/roberta-fa-zwnj-base-ner"  # Roberta
model_name_or_path = "HooshvareLab/distilbert-fa-zwnj-base-ner"  # Distilbert
# model_name_or_path = "HooshvareLab/albert-fa-zwnj-base-v2-ner"  # Albert

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)  # Pytorch
# model = TFAutoModelForTokenClassification.from_pretrained(model_name_or_path)  # Tensorflow

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "در سال ۲۰۱۳ درگذشت و آندرتیکر و کین برای او مراسم یادبود گرفتند."

ner_results = nlp(example)
print(ner_results)
```

## Models

### Hugging Face Model Hub

- [Bert](https://huggingface.co/HooshvareLab/bert-fa-zwnj-base-ner)
- [Roberta](https://huggingface.co/HooshvareLab/robert-fa-zwnj-base-ner)
- [Distilbert](https://huggingface.co/HooshvareLab/distilbert-fa-zwnj-base-ner)
- [Albert](https://huggingface.co/HooshvareLab/albert-fa-zwnj-base-v2-ner)

### Training
All models were trained on a single NVIDIA P100 GPU with following parameters.

**Arguments**
```bash
"task_name": "ner"
"model_name_or_path": model_name_or_path
"train_file": "/content/ner/train.csv"
"validation_file": "/content/ner/valid.csv"
"test_file": "/content/ner/test.csv"
"output_dir": output_dir
"cache_dir": "/content/cache"
"per_device_train_batch_size": 16
"per_device_eval_batch_size": 16
"use_fast_tokenizer": True
"num_train_epochs": 15.0
"do_train": True
"do_eval": True
"do_predict": True
"learning_rate": 2e-5
"evaluation_strategy": "steps"
"logging_steps": 1000
"save_steps": 1000
"save_total_limit": 2
"overwrite_output_dir": True
"fp16": True
"preprocessing_num_workers": 4
```


## Cite
Please cite this repository in publications as the following:

```bibtext
@misc{ParsNER,
  author = {Hooshvare Team},
  title = {Pretrained model for NER in Farsi},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hooshvare/parsner}},
}
```


## Questions?
Post a Github issue on the [ParsNER Issues](https://github.com/hooshvare/parsner/issues) repo.