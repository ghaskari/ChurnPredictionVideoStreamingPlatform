# Churn Prediction Video Streaming Platform

This project predicts customer churn for a video streaming platform. Churn prediction is a crucial task for subscription-based businesses, allowing them to identify customers who are likely to cancel their subscriptions. This project uses a dataset of customer subscriptions from 2021 to build and evaluate churn prediction models.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Customization and Extension](#customization-and-extension)

---

## Project Overview
The project aims to predict customer churn in a video streaming platform using classical machine learning models. The dataset contains customer subscription information, including both numerical and categorical features, along with the target variable, Churn.

**Key Components:**
- Exploratory Data Analysis (EDA) to understand the dataset.
- Feature engineering and preprocessing to prepare data for modeling.
- Model training and evaluation to find the best model for churn prediction.
- Feature importance analysis to understand the most influential features.

## Getting Started
To set up and run the project, follow these steps:

1. **Clone the Project Repository**
   ```bash
   git clone https://github.com/ghaskari/ChurnPredictionVideoStreamingPlatform.git
   ```

2. **Install Required Libraries**
   Navigate to the project directory and install the dependencies using the provided `requirements.txt`:
   ```bash
   cd ChurnPredictionVideoStreamingPlatform
   pip install -r requirements.txt
   ```

## Project Structure
- **`data/`**: Contains the dataset files used in this project, such as training and test data.
- **`scripts/`**: Contains code for data analysis, model development, and evaluation.
- **`requirements.txt`**: Lists the required Python libraries.
- **`README.md`**: This guide explaining the project and how to use it.

## Usage Guide
To run the project, follow these steps:

1. **Data Exploration**
   - Load the dataset and perform exploratory data analysis (EDA) to understand its structure.
   - Visualize features using bar plots and heatmaps to identify key relationships.

2. **Feature Engineering and Preprocessing**
   - Apply one-hot encoding to categorical features.
   - Handle missing values and ensure consistent data types.

3. **Model Development**
   - Train classical machine learning models, such as Random Forest, Gradient Boosting, and Support Vector Machines.
   - Use GridSearchCV for hyperparameter tuning to optimize model performance.
   - Compare different models to determine which performs best.

4. **Model Evaluation and Feature Importance**
   - Evaluate models with metrics like accuracy, precision, recall, F1-score, and confusion matrices.
   - Identify the best-performing model based on evaluation metrics.
   - Use feature importance plots to understand which features are most significant for churn prediction.

## Customization and Extension
Here are some ways you can extend or customize the project:

- **Experiment with Different Models**
  - Try other machine learning models or ensemble methods to improve accuracy.
  - Fine-tune hyperparameters for better model performance.

- **Analyze Feature Importance**
  - Use the feature importance plots to identify the most impactful features.
  - Consider removing less significant features to reduce model complexity.

- **Additional Data Insights**
  - Explore specific subsets of the data to understand customer behavior in more detail.
  - Investigate potential relationships and correlations that could impact churn prediction.
