# Research Paper Classification (Cancer vs Non-Cancer)

This repository contains an **end-to-end machine learning pipeline** to classify research paper abstracts into two categories:
- **Cancer**
- **Non-Cancer**

The pipeline uses:
- **DistilBERT** for baseline classification.
- **LoRA (Low-Rank Adaptation)** for fine-tuning with fewer trainable parameters.
- **FastAPI** for exposing a REST API for inference.

---

## **Project Features**
1. **Dataset Preparation** – Cleans text data and splits into train/val/test.
2. **Baseline Training** – Trains a DistilBERT-based model.
3. **LoRA Fine-tuning** – Applies parameter-efficient fine-tuning.
4. **Evaluation & Comparison** – Compares baseline vs fine-tuned performance.
5. **Inference** – Outputs predictions in JSON format with confidence scores.
6. **FastAPI REST API** – Real-time prediction endpoint with Swagger UI.

---

## **Project Structure**
research-paper-classification/
│
├── src/
│ ├── data_prep.py # Dataset cleaning & splitting
│ ├── train.py # Baseline DistilBERT training
│ ├── finetune_lora.py # LoRA fine-tuning
│ ├── evaluate.py # Evaluation & confusion matrix
│ ├── compare_models.py # Compare baseline vs LoRA
│ ├── inference.py # Run predictions and save JSON
│ └── api.py # REST API using FastAPI
│
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/rdiksha616/research-paper-classification.git
cd research-paper-classification

2. Install Dependencies

pip install -r requirements.txt

Usage
Step 1: Prepare the Dataset

Ensure you have a dataset structured as:

Dataset/
├── Cancer/
│   ├── paper1.txt
│   ├── paper2.txt
│   ...
└── Non-Cancer/
    ├── paper1.txt
    ├── paper2.txt
    ...

Then run:

python src/data_prep.py --dataset_root Dataset --out_dir data

Step 2: Train Baseline Model

python src/train.py --data_dir data/processed --output_dir models/baseline

Step 3: Fine-tune with LoRA

python src/finetune_lora.py --data_dir data/processed --model_name distilbert-base-uncased --output_dir models/finetuned_lora

Step 4: Evaluate Models

python src/evaluate.py --data_dir data/processed --model_dir models/baseline
python src/evaluate.py --data_dir data/processed --model_dir models/finetuned_lora

Step 5: Compare Baseline vs LoRA

python src/compare_models.py

Example Output:

Baseline Accuracy : 0.96
Fine-tuned Accuracy: 0.93
Accuracy Improvement: -0.03

Step 6: Run Inference

python src/inference.py --model_dir models/finetuned_lora \
    --input_text "This study focuses on lung cancer treatment." \
    --output_json predictions.json

Sample Output (predictions.json):

[
  {
    "abstract_id": "manual_input",
    "predicted_labels": ["Cancer"],
    "confidence_scores": {"Cancer": 0.92, "Non-Cancer": 0.08},
    "extracted_diseases": ["Lung Cancer"]
  }
]

Step 7: Start FastAPI Server

python src/api.py

    Open http://127.0.0.1:8000/docs in your browser for Swagger UI.