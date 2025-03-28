# Fake News Detection Pipeline

## Overview

This project processes a fake news corpus to build machine learning models that classify news articles as reliable or fake. It includes comprehensive text preprocessing, exploratory data analysis, simple modeling (logistic regression), advanced modeling (SVM, Naive Bayes, Gradient Boosting, Random Forest), and evaluates performance using an external dataset (LIAR).

## Project Structure

```
├── Datasets
│   ├── news_sample.csv
│   ├── 995000_rows.csv
│   ├── article_texts.csv
│   ├── train.tsv
│   ├── valid.tsv
│   └── test.tsv
├── models
│   └── (trained model and pipeline files will be saved here)
├── README.md
└── fake_news.ipynb  # or separate scripts for each step
```

## Requirements

- **Python Version:** 3.7 or higher

**Libraries:**
- pandas
- numpy
- nltk
- matplotlib
- seaborn
- scikit-learn
- joblib
- cleantext
- modin[ray]
- ray

Create a `requirements.txt`:

```text
pandas
numpy
nltk
matplotlib
seaborn
scikit-learn
joblib
cleantext
modin[ray]
ray
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/Feros47/GDS-Exam.git
cd GDS_Exam
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK Resources

The necessary NLTK datasets (`punkt`, `stopwords`) will be downloaded automatically on first execution.

## Datasets

Ensure the following datasets are located in the `Datasets` directory:
- `news_sample.csv`: Fake news corpus sample.
- `995000_rows.csv`: Large-scale dataset (~995,000 rows).
- `article_texts.csv`: Reliable articles (e.g., from BBC).
- `train.tsv`, `valid.tsv`, `test.tsv`: LIAR dataset splits.

## Usage

### 1. Preprocessing and Exploration

This step includes:
- Text cleaning (dates, URLs, emails, numbers removal)
- Tokenization, stopword removal, stemming
- Exploratory plots and vocabulary statistics
- Scalable processing with Modin and Ray


### 2. Simple Model (Logistic Regression)

- Bag-of-words features (CountVectorizer)
- Logistic regression training and evaluation


### 3. Advanced Models (Grid Search and TF-IDF)

- TF-IDF features (uni-gram and bi-gram)
- Models: SVM, Naive Bayes, Logistic Regression, Gradient Boosting, Random Forest
- Hyperparameter tuning via grid search
- Model evaluation and confusion matrices


### 4. LIAR Dataset Evaluation

- Text processing and binary label mapping
- Evaluation of simple and advanced models on LIAR dataset
- Plotting confusion matrices


## Output and Model Artifacts

- **Models**: Trained pipelines and classifiers saved as `.joblib` in the `models` directory.
- **Plots**: Word frequency, domain distributions, confusion matrices.
- **Logs**: Console outputs include training time, accuracy, F1 scores, and optimal hyperparameters.

## Notes

- **Scalability**: Utilizes Modin and Ray; ensure adequate computational resources.
- **Reproducibility**: Random states (`random_state=42`) are set.
- **Customization**: Modify feature extraction parameters and grid search configurations as required.
