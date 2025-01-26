# NLP-Emotion-Classification

## Project Overview
This repository contains the implementation of a text classification project for identifying emotions (e.g., joy, sadness, fear, anger) in tweet texts. The project explores three deep learning approaches:
1. **Fully Connected Neural Network (FCNN)**
2. **Recurrent Neural Network (RNN)** with LSTM/GRU
3. **Fine-tuned Transformer Model** (e.g., BERT from HuggingFace)

The dataset consists of labeled tweet texts split into `train.txt`, `test.txt`, and `validation.txt`. The goal is to compare the performance of these models and analyze their effectiveness for emotion classification.

This project was developed as part of an NLP Exam by **Alessandro Mencarelli**, **Mauricio Rodriguez**, and **Mario Zuna**.

---

## Repository Structure
```
NLP-Emotion-Classification/
│
├── data/
│   ├── train.txt
│   ├── test.txt
│   └── validation.txt
│
├── notebooks/
│   └── Emotion_Classification_Notebook.ipynb
│
├── models/
│   ├── fcnn_model.h5
│   ├── rnn_model.h5
│   └── transformer_model/
│
├── README.md
└── requirements.txt
```

---

## Notebook
The main notebook, **`Emotion_Classification_Notebook.ipynb`**, contains:
- **Data Preprocessing**: Tokenization, padding, and encoding of tweet texts.
- **Model Implementation**:
  - Fully Connected Neural Network (FCNN)
  - Recurrent Neural Network (RNN) with LSTM/GRU
  - Fine-tuned Transformer Model (BERT)
- **Results Comparison**: Analysis of model performance and insights.

---

## Requirements
To run the notebook, install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/menca-lsn/NLP-Emotion-Classification.git
   cd NLP-Emotion-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook notebooks/Emotion_Classification_Notebook.ipynb
   ```

4. Follow the instructions in the notebook to preprocess the data, train the models, and evaluate their performance.

---

## Dataset
The dataset consists of tweet texts labeled with emotions (e.g., joy, sadness, fear, anger). It is split into:
- **Train set**: Used for training the models.
- **Test set**: Used for evaluating model performance during development.
- **Validation set**: Used for final performance evaluation to ensure no overfitting.

---

## Models
1. **Fully Connected Neural Network (FCNN)**:
   - A simple feedforward neural network for baseline performance.

2. **Recurrent Neural Network (RNN)**:
   - Uses LSTM/GRU layers to capture sequential dependencies in text data.

3. **Transformer Model (BERT)**:
   - Fine-tunes a pretrained BERT model from HuggingFace for state-of-the-art performance.

---

## Results
- **FCNN**: Achieved **0.82 accuracy**.
- **RNN**: Achieved **0.89 accuracy**.
- **Transformer (BERT)**: Achieved **0.92 accuracy**, leveraging pretrained language representations for the best performance.

---

Happy coding! 🚀
