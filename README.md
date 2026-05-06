# Fake News Detection System

## Overview
This project is an AI-powered Fake News Detection System developed for the AI Systems Engineering module. The system utilizes both Classical Machine Learning algorithms and modern Deep Learning approaches (DistilBERT) to classify news articles as authentic or fake.

## Team Contributions (Group 4)
- **(A00068444)**: Data Cleaning & Preprocessing (EDA, text normalization, tokenization)
- **(A00075730)**: Classical ML Models (TF-IDF Feature Engineering, Logistic Regression, Random Forest)
- **(A00084866)**: Deep Learning (PyTorch setup, DistilBERT tokenization, and fine-tuning)
- **(A00082648)**: Evaluation & Visualization (Confusion matrices, ROC curves, comparative analysis)

## Technologies Used
- **Language:** Python 3
- **Classical ML:** Scikit-learn (Logistic Regression, Random Forest, TF-IDF)
- **Deep Learning:** PyTorch, Hugging Face Transformers (DistilBERT)
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **NLP Utilities:** NLTK

## Project Structure
- `Group4.ipynb`: The main Jupyter Notebook containing the end-to-end pipeline.
- `FakeNewsSet.csv`: The dataset containing real and fake news articles.
- `requirements.txt`: List of required Python dependencies.

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/A00068444/Group_4.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook using Jupyter:
   ```bash
   jupyter notebook
   ```
4. Run the notebook cells sequentially to execute data preprocessing, train the models, and view the evaluation metrics.

## Results & Evaluation
A comprehensive comparative analysis was conducted between traditional TF-IDF-based models and the transformer-based DistilBERT model. Detailed evaluation charts, including confusion matrices and classification reports, can be viewed directly within the notebook outputs.
