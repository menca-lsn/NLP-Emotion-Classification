# NLP Course Graded Project - Spring 2024

## Project Overview

This project is part of the NLP course for Spring 2024. The goal is to build and compare different machine learning models for text classification based on emotion labels. The dataset consists of tweet texts labeled with emotions such as joy, sadness, fear, and anger. The dataset is split into train, test, and validation sets. Initially, only the train and test sets are provided, with the validation set to be released at the end of the project to evaluate the final performance of the models.

## Project Goals

1. **Train Different Classification Models:**
   - **Fully Connected Neural Network (FCNN)**
   - **Recurrent Neural Network (RNN)** based on LSTM or GRU
   - **Fine-tuned Transformer Architecture** from a pretrained model (e.g., from HuggingFace)

2. **Compare and Analyze Models:**
   - Compare the results of the different models.
   - Analyze and explain the differences in performance.
   - Discuss what you have learned from this exercise and how you would approach another text classification use case.

## Dataset

The dataset contains tweet texts with corresponding emotion labels. The dataset is divided into:
- **Train set**: Used to train the models.
- **Test set**: Used to evaluate the models during development.
- **Validation set**: Provided at the end to check the final performance and ensure no overfitting on the test data.

## Repository Structure

```
NLP_Project/
│
├── data/
│   ├── train.txt
│   ├── test.txt
│   └── validation.txt
│
├── notebooks/
│   ├── Data_Preprocessing.ipynb
│   ├── FCNN_Model.ipynb
│   ├── RNN_Model.ipynb
│   ├── Transformer_Model.ipynb
│   └── Model_Comparison.ipynb
│
├── models/
│   ├── fcnn_model.h5
│   ├── rnn_model.h5
│   └── transformer_model/
│
├── README.md
└── requirements.txt
```

## Requirements

To run the notebooks and train the models, you will need the following Python packages:

- TensorFlow
- PyTorch
- Transformers (HuggingFace)
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Seaborn

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/NLP_Project.git
   cd NLP_Project
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebooks in the `notebooks/` directory to preprocess the data, train the models, and compare their performance.

4. Submit your final project report and code by the deadline.
