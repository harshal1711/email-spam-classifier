# Spam Email Classification with Cost-Sensitive Modeling

This project focuses on building classification models to distinguish between spam and non-spam emails using the [UCI Spambase dataset](https://archive.ics.uci.edu/ml/datasets/Spambase). We train and compare multiple models under two different objectives:
- Maximizing predictive accuracy
- Minimizing misclassification cost (with a 10:1 cost ratio)

---

## Dataset Overview

- **Records**: 4,601 email messages
- **Features**: 57 attributes derived from message content (e.g., word frequency, punctuation, capital letter usage)
- **Target**: `1` (spam), `0` (non-spam)

---

## Classification Modeling Summary

### Part (A): Optimizing for Accuracy

- Models evaluated using **nested cross-validation**:
  - Logistic Regression, KNN, Decision Tree, SVC, Neural Network, LightGBM
- Performance measured via **Accuracy** and **AUC**

| Model              | Mean Accuracy | Std Accuracy | AUC      |
|-------------------|---------------|---------------|----------|
| LogisticRegression| 0.9291        | 0.0122        | 0.9708   |
| KNN               | 0.9111        | 0.0085        | 0.9614   |
| DecisionTree      | 0.9076        | 0.0064        | 0.9216   |
| SVC               | 0.9310        | 0.0107        | 0.9717   |
| NeuralNet         | 0.9353        | 0.0106        | 0.9765   |
| **LightGBM**      | **0.9538**    | **0.0052**    | **0.9884** |

- **Best Accuracy Model**: LightGBM
  - Test Accuracy: **95.01%**
  - Confusion Matrix:
    ```
    [[543  18]
     [ 28 332]]
    ```

---

### Part (B): Cost-Sensitive Classification (10:1 Misclassification Cost)

- Models evaluated using **nested cross-validation**
- Custom cost function used to weight **false negatives 10× higher** than false positives

| Model              | Mean Cost | Std Cost | AUC      |
|-------------------|-----------|----------|----------|
| LogisticRegression| 314.2     | 66.98    | 0.9702   |
| KNN               | 310.0     | 29.91    | 0.9633   |
| DecisionTree      | 350.4     | 33.60    | 0.9191   |
| SVC               | 253.4     | 40.74    | 0.9755   |
| NeuralNet         | 237.6     | 55.14    | 0.9757   |
| **LightGBM**      | **191.0** | **35.75**| **0.9879** |

- **Best Cost-Sensitive Model**: LightGBM
  - Test Accuracy: **95.44%**
  - Test Cost: **231**
  - Confusion Matrix:
    ```
    [[540  21]
     [ 21 339]]
    ```

---

## Key Takeaways

- **LightGBM consistently outperformed** other models in both standard and cost-sensitive settings.
- **Neural Networks and SVC** were strong contenders, especially in AUC performance.
- Cost-sensitive modeling significantly improved performance on **false negative reduction**, which is critical in spam detection.
