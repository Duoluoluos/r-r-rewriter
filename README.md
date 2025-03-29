# Read the Docs Before Rewriting: Equip Rewriter with Domain Knowledge via Continual Pre - training

Before delving into the technical details, it's crucial to understand the significance of this project. By equipping a rewriter with domain knowledge through continual pre-training, we aim to enhance its performance in handling specific tasks more effectively.

## Installation
1. **Create and activate a conda environment**:
   - Open your terminal and run the following commands:
   ```bash
   conda create -n rr_rewriter python=3.11
   conda activate rr_rewriter
   ```

2. **Install the necessary dependencies**:
   - After activating the environment, install the required packages using the following command:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key**:
   - To use the OpenAI - related features, you need to set your API key. In the terminal, run:
   ```bash
   export OPENAI_API_KEY = <your_api_key>
   ```
   Replace `<your_api_key>` with your actual OpenAI API key.

## Continual Pretraining (CPT)
- **Data location and generation**:
  - The example CPT document data for Professional QA is stored in the `./data` directory. But due to confidentiality requirements, applications for SRCQA's document data should be emailed to XXX(anonymized for review). Applications for FintextQA's document data should be emailed to XXX(anonymized for review).
  - General QA document data for CPT can be referred to (https://github.com/GermanT5/wikipedia2corpus)
  - If you want to generate data specifically for CPT, you can use the codes provided in `./data_generator/llm_annotator.ipynb`.
- **Training process**:
  - We utilize the (https://github.com/hiyouga/LLaMA-Factory).
  - To fine - tune hyperparameters, it is highly recommended to use the webui interface. You can start the webui by running the following command in the terminal:
  ```bash
  llamafactory-cli webui
  ```
  - The training parameters can be referred to in `data/training_args_CPT.yaml`. This file contains all the necessary settings for the CPT process, such as learning rate, batch size, and number of training epochs.

## Supervised Finetuning (SFT)
- **Data location**:
  - The sample SFT data, including `SRC_SFT_P.json`, `Syllabus_SFT_P.json`, and `Fintext_SFT_P.json`, are all located in the `./data` directory.
  - Similar to CPT data generation, you can use the codes in `./data_generator/llm_annotator.ipynb` to generate annotation data for SFT.
- **Training parameters**:
  - The training parameters for SFT can be found in `data/training_args_SFT.yaml`. These parameters are crucial for optimizing the SFT process and achieving better performance.

## Main Results
To evaluate the performance of the R&R - 7B model, you can execute the following commands:
- **For SRCQA**:
  ```bash
  python SRC_test.py
  ```
- **For FintextQA**:
  ```bash
  python Fintext_test.py
  ```
- **For SyllabusQA**:
  ```bash
  python Syllabus_test.py
  ```

- **For AmbigQA**
  ```bash
  # generate predictions
  python Ambig_test.py
  # evaluate predictions
  python Ambig_evaluate.py --reference_path  <path> --prediction_path  <path>
  ```
  Ground truth for AmbigQA can be downloaded through (https://nlp.cs.washington.edu/ambigqa/data/ambignq_with_evidence_articles.zip)
If you want to test a different model type, you can easily modify the settings in the `config.yaml` file. 