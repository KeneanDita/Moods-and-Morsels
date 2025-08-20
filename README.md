# Moods and Morsels

## Project Overview

This project performs **sentiment analysis** on a subset of the Amazon Fine Food Reviews dataset. The goal is to classify reviews as **positive** or **negative**, helping businesses and researchers gain insights into customer satisfaction through natural language processing (NLP) techniques.

The original dataset contained **over 565,000 reviews**, but due to resource constraints and to accelerate model prototyping, it was **reduced to 10,000 records** for this project.

---

## 📊 Dataset Information

* **Source**: [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
* **Original Size**: \~568,000 reviews
* **Subset Used**: 10,000 reviews (random sample)
* **Features**:

  * `Text`: The body of the customer review
  * `Score`: Rating from 1 to 5
* **Target Variable**:

  * Transformed into binary sentiment:

    * `Positive` (Score ≥ 4)
    * `Negative` (Score ≤ 2)
    * Score = 3 reviews were removed for clarity

---

## Project Structure

```
amazon-food-review-sentiment
├── data/
│   └── reviews_sample.csv
├── M&M.ipynb  # Notebook for training and viewing model performance
├── models/
│   └── transformers model folder
├── UI.py #run this streamlit UI
├── README.md
└── requirements.txt
```

---

## Requirements and Usage

* Python 3.8+
* pandas, numpy
* scikit-learn
* nltk
* transformers
* matplotlib
* seaborn

Step 1. install requirements:

Install via:

```bash
pip install -r requirements.txt
```

Step 2:

~ Make sure you have the transformers model downloaded and it get's stored in a folder named 'Models'.

run this after the 2 steps are done

Step 3:

```bash
Streamlit run .\UI.py
```

Enjoy testing the program. and give it a star ⭐ to appreciate the effort.

### Author:  [KeneanDita](https://github.com/keneandita/)