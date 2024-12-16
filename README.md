## Installation

First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/Duoluoluos/R-R-Rewriter
```
Create and activate a conda environment


```bash
conda create -n rr_rewriter python=3.11
conda activate rr_rewriter
```


Install the necessary dependencies.


```bash
pip install -r requirements.txt
```


## Continual Pretraining(CPT)
The example CPT data is stored in the directory `database/src_document/json_files/doc`. You can leverage the codes provided in `data_generator/llm_annotator.ipynb` to generate data specifically for CPT.

For continual pretraining of the LLM rewriter, we rely on the [llamafactory project](https://github.com/hiyouga/LLaMA-Factory). To fine-tune hyperparameters, we highly recommend utilizing the webui interface by running the following command:

```bash
llamafactory-cli webui
```

## Supervised Finetuning(SFT)
The question-answer pairs are stored in the directory `database/src_document/json_files/test/mock`. You can leverage the codes provided in `data_generator/llm_annotator.ipynb` to generate annotation data for SFT.


## Main Results
To evaluate the performance of the R&R-7B model on SRCQA, simply execute the following command:
```bash
python SRT_test.py
```
If you wish to test a different model type, you can conveniently modify the settings in the `config.yaml` file.