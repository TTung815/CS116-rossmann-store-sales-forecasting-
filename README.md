# Rossmann Store Sales Forecasting

Machine learning project for forecasting daily sales of Rossmann drug stores using historical sales, store metadata, promotion information, holidays, and time-based patterns.

This project was developed for the CS116 - Python for Machine Learning course and submitted to the Kaggle Rossmann Store Sales competition.

## Highlights

- Built an end-to-end regression pipeline for a real-world retail time-series dataset.
- Performed exploratory data analysis on sales patterns, holidays, promotions, store types, and seasonality.
- Engineered store-level, promotion, competition, and cyclical date features.
- Tuned XGBoost, LightGBM, and CatBoost with Optuna.
- Improved performance using Voting and Stacking ensemble models.
- Achieved a reported Kaggle ranking of Top 79 / 3300, approximately top 2.4%.

## Problem

Rossmann operates over 1,000 stores. The goal is to predict future daily sales for each store based on:

- Historical sales data
- Store metadata
- Promotion campaigns
- Public and school holidays
- Store opening status
- Competition distance and competition opening date
- Calendar and seasonal effects

## Dataset

The project uses the Kaggle Rossmann Store Sales dataset:

- `train.csv`: historical daily sales records
- `test.csv`: future store-day records for prediction
- `store.csv`: store-level metadata

The raw dataset is not included in this repository because it should be downloaded from Kaggle.

Competition page: https://www.kaggle.com/c/rossmann-store-sales

## Methodology

### 1. Exploratory Data Analysis

Key findings:

- `Sales` is right-skewed and contains high-value outliers.
- Stores are usually closed on Sundays and major holidays.
- Promotions are strongly associated with higher sales.
- `Customers`, `Open`, and `Promo` are highly related to `Sales`.
- Sales show seasonal peaks around March, summer months, and the end of the year.

### 2. Preprocessing

- Merged sales data with store metadata.
- Filled missing competition and promotion fields using meaningful sentinel values.
- Filled missing `Open` values in the test set.
- Removed invalid rows where stores were open but had zero sales.
- Applied log transformation to the target variable.
- Used robust scaling for skewed numerical features.

### 3. Feature Engineering

Created features such as:

- Average sales per store per day
- Average customers per store per day
- Sales per customer per store
- Time since Promo2 started
- Competition age
- Cyclical date encodings using sine and cosine
- Distance to seasonal sales peaks such as March, July, and December

### 4. Modeling

Models evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- XGBoost
- CatBoost
- LightGBM
- Voting Regressor
- Stacking Regressor

Hyperparameter tuning:

- Grid search for linear models
- Optuna for boosting models

## Results

| Model | MAE | RMSPE |
| --- | ---: | ---: |
| Linear Regression | 0.1549 | 0.0254 |
| Ridge Regression | 0.1549 | 0.0253 |
| Lasso Regression | 0.1553 | 0.0253 |
| XGBoost | 0.1137 | 0.0185 |
| CatBoost | 0.1143 | 0.0183 |
| LightGBM | 0.1114 | 0.0180 |
| Voting Regressor | 0.1100 | 0.0179 |
| Stacking Regressor | 0.1093 | 0.0177 |

Reported Kaggle result:

- Public score: 0.10763
- Private score: 0.11286
- Rank: 79 / 3300

## Repository Structure

```text
.
├── notebooks/
│   └── CS116_code.ipynb
├── reports/
│   ├── CS116_Report.pdf
│   └── CS116_Final_Slide.pdf
├── data/
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

The notebook was originally developed in Google Colab with GPU support. It now supports two data-loading modes:

### Option 1: Run Locally

1. Download the dataset from Kaggle.
2. Place the raw CSV files under `data/raw/`:

```text
data/raw/
├── train.csv
├── test.csv
└── store.csv
```

3. Install the common Python dependencies:

```bash
pip install -r requirements.txt
```

4. Start Jupyter from the repository root and open the notebook:

```bash
jupyter notebook
```

5. Run:

```text
notebooks/CS116_code.ipynb
```

The notebook detects `data/raw/` automatically when the files exist. If you run it from inside the `notebooks/` folder, it also checks the parent repository folder.

Note: the EDA, preprocessing, and feature engineering sections can run in a normal local Python environment after installing the dependencies. The model training section uses RAPIDS cuML and GPU-specific settings, so a CPU-only local machine may require replacing cuML estimators with scikit-learn equivalents.

### Option 2: Run on Google Colab

1. Upload `notebooks/CS116_code.ipynb` to Google Colab.
2. Enable GPU runtime:

```text
Runtime > Change runtime type > GPU
```

3. Run the notebook from the first cell.

If `data/raw/` is not found, the first notebook cell downloads and unzips the dataset into:

```text
/content/rossmann-store-sales/
```

This matches the original Colab workflow used for the project.

## Project Artifacts

- Full report: `reports/CS116_Report.pdf`
- Presentation slides: `reports/CS116_Final_Slide.pdf`
- Main notebook: `notebooks/CS116_code.ipynb`
