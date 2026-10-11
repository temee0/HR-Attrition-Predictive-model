# HR Employee Attrition Prediction

## Project Overview

This project explores employee attrition using machine learning, with the goal of predicting whether an employee is likely to leave an organisation.

The project was developed as part of my machine learning learning journey. Rather than focusing only on achieving a high model score, I used the project to understand the full classification workflow, including exploratory data analysis, feature selection, categorical encoding, train-test splitting, Logistic Regression, Decision Trees, confusion matrices, overfitting, and model evaluation.

The target variable is `left`:

- `0` = Employee stayed
- `1` = Employee left

The project started with Logistic Regression and was later extended to a Decision Tree classifier as I progressed in my machine learning studies.

---

## Dataset

The dataset contains 14,999 employee records and the following variables:

| Feature | Description |
|---|---|
| `satisfaction_level` | Employee satisfaction level |
| `last_evaluation` | Employee's most recent evaluation score |
| `number_project` | Number of projects assigned |
| `average_montly_hours` | Average monthly working hours |
| `time_spend_company` | Number of years spent at the company |
| `Work_accident` | Whether the employee had a workplace accident |
| `promotion_last_5years` | Whether the employee was promoted in the last five years |
| `Department` | Employee department |
| `salary` | Salary category: low, medium, or high |
| `left` | Target variable indicating whether the employee left |

There were no missing values in the dataset.

---

## Exploratory Data Analysis

Before building the models, I explored how the available features related to employee attrition.

### Numeric Correlation

The strongest linear relationship with attrition was employee satisfaction.

Some of the correlations with `left` were:

| Feature | Correlation |
|---|---:|
| `satisfaction_level` | -0.388 |
| `Work_accident` | -0.155 |
| `time_spend_company` | 0.145 |
| `average_montly_hours` | 0.071 |
| `promotion_last_5years` | -0.062 |
| `number_project` | 0.024 |
| `last_evaluation` | 0.007 |

The negative correlation for satisfaction suggests that employees with higher satisfaction levels were less likely to leave.

However, this analysis also showed why correlation should not be used alone for feature selection.

For example, `number_project` had a very low overall correlation with attrition, but grouping employees by project count revealed a much stronger non-linear pattern.

### Attrition by Number of Projects

| Projects | Attrition Rate |
|---:|---:|
| 2 | 65.6% |
| 3 | 1.8% |
| 4 | 9.4% |
| 5 | 22.2% |
| 6 | 55.8% |
| 7 | 100% |

Employees with either very few or very many projects had much higher attrition rates.

This relationship would be difficult to understand from the correlation value alone.

### Attrition by Time at Company

The highest attrition rate occurred around five years of employment.

| Years at Company | Attrition Rate |
|---:|---:|
| 2 | 1.6% |
| 3 | 24.6% |
| 4 | 34.8% |
| 5 | 56.6% |
| 6 | 29.1% |
| 7 | 0% |
| 8 | 0% |
| 10 | 0% |

This was another example of a non-linear relationship in the dataset.

### Salary and Attrition

Salary showed a clear relationship with employee attrition:

| Salary | Attrition Rate |
|---|---:|
| Low | 29.7% |
| Medium | 20.4% |
| High | 6.6% |

Employees in the low salary category had the highest attrition rate, while employees with high salaries had the lowest.

### Department and Attrition

Attrition rates also varied across departments.

HR had the highest attrition rate at approximately 29%, while Management and Research & Development had the lowest rates at approximately 14–15%.

Department was initially excluded from some models so that its contribution could later be tested separately.

---

## Data Preparation

The categorical variables `salary` and `Department` could not be passed directly into the machine learning models because they contained text.

They were converted into numerical features using pandas dummy variables

The target variable remained:

```python
y = Data['left']
```

while the independent variables were tested in different combinations.

The data was divided into training and testing sets using `train_test_split`.

A fixed `random_state` was used during later experiments so that model comparisons were performed on the same train-test split.

---

## Logistic Regression

The first classification algorithm used in the project was Logistic Regression.

Several combinations of features were tested to understand how the independent variables affected model performance.

Some of the results included:

| Features | Accuracy |
|---|---:|
| Satisfaction + Salary | 76.4% |
| Satisfaction + Department + Salary | 76.9% |
| Satisfaction + Promotion + Salary | 76.5% |
| Satisfaction + Promotion + Hours + Accident + Salary | 76.8% |
| Satisfaction + Promotion + Hours + Accident + Salary + Department | **77.2%** |

The highest Logistic Regression accuracy obtained was approximately **77.2%**.

However, accuracy alone gave an incomplete picture.

The dataset was imbalanced:

- Approximately 76.5% of employees stayed.
- Approximately 23.5% of employees left.

This meant that a model predicting that nearly everyone would stay could still achieve a relatively high accuracy score.

### Confusion Matrix Analysis

One Logistic Regression model produced:

```text
[[2141  153]
 [ 544  162]]
```

This meant:

- 2,141 employees who stayed were correctly classified.
- 153 employees who stayed were incorrectly predicted to leave.
- 544 employees who left were incorrectly predicted to stay.
- Only 162 employees who left were correctly identified.

Out of 706 actual leavers, only 162 were detected.

This represented a recall of roughly **23% for employee attrition**.

This became an important learning point in the project:

> A model can have acceptable overall accuracy while still performing poorly on the class that matters most.

---

## Feature Experimentation

Different combinations of independent variables were tested using Logistic Regression.

For example, removing `average_montly_hours` slightly improved accuracy in one experiment, while removing `Work_accident` caused the score to fall again.

Department was also tested separately.

Adding Department produced a small improvement in accuracy, showing that it contributed some predictive information even though department attrition rates initially appeared relatively similar.

These experiments helped demonstrate that feature selection should not rely only on simple correlation.

---

## Decision Tree Model

The next stage of the project introduced a Decision Tree classifier.

Unlike Logistic Regression, Decision Trees can naturally capture more complex and non-linear relationships.

This was particularly relevant for variables such as:

- `number_project`
- `time_spend_company`

where the attrition pattern was not linear.

The Decision Tree was trained using a broader set of available features.

### Initial Decision Tree

The first Decision Tree achieved approximately:

- **100% training accuracy**
- **96% testing accuracy**

Although the test performance was strong, the perfect training score indicated that the model was overfitting.

### What is Overfitting?

Overfitting happens when a model learns the training data too closely.

Instead of learning only the general patterns in the data, it may also learn noise or very specific details from the training set.

This can cause the model to perform extremely well on data it has already seen while performing worse on new, unseen data.

---

## Reducing Overfitting

The Decision Tree was adjusted to reduce its complexity.

After limiting the tree, the results improved to:

- **Training Accuracy: 97.57%**
- **Testing Accuracy: 97.06%**

The difference between the training and testing scores was only around 0.5 percentage points.

This suggested that the model was generalising much better to unseen data.

### Final Confusion Matrix

The improved Decision Tree produced:

```text
[[9000  134]
 [ 219 2646]]
```

This means:

- **9,000** employees who stayed were correctly predicted to stay.
- **134** employees who stayed were incorrectly predicted to leave.
- **219** employees who left were incorrectly predicted to stay.
- **2,646** employees who left were correctly identified.

Out of 2,865 employees who actually left, the model correctly identified 2,646.

This gives an attrition recall of approximately **92.4%**.

The model also achieved approximately **95.2% precision** when predicting that an employee would leave.

---

## Model Progress

The improvement throughout the project can be summarised as follows:

| Stage | Result |
|---|---|
| Logistic Regression | ~76–77% accuracy |
| Logistic Regression attrition recall | ~23–24% |
| Initial Decision Tree | ~96% test accuracy |
| Initial Decision Tree training accuracy | 100% — overfitting |
| Improved Decision Tree | ~97.1% test accuracy |
| Improved Decision Tree training accuracy | ~97.6% |
| Final attrition recall | ~92.4% |

The most important improvement was not simply the increase from 77% to 97% accuracy.

The biggest improvement was the model's ability to correctly identify employees who actually left.

The earlier Logistic Regression model detected only around one-quarter of employee leavers, while the improved Decision Tree detected more than 92%.

---

## Key Lessons

This project helped reinforce several important machine learning concepts:

- `.fit()` is the stage where the model learns from the training data.
- Accuracy alone is not always enough to evaluate a classification model.
- Class imbalance can make accuracy misleading.
- Confusion matrices provide much more information about classification performance.
- Features with low linear correlation can still be highly useful.
- Logistic Regression and Decision Trees learn patterns differently.
- Decision Trees can capture non-linear relationships more naturally.
- A model with 100% training accuracy is not automatically a good model.
- Training and testing performance should be compared to identify overfitting.
- Feature selection is best treated as an experiment rather than relying on one statistical measure.
- Keeping the same train-test split makes model comparisons more reliable.

---

## Final Model

The final selected model is a Decision Tree classifier with approximately:

```text
Training Accuracy: 97.57%
Testing Accuracy:  97.06%
Attrition Recall:  92.4%
Attrition Precision: 95.2%
```

The close training and testing results indicate good generalisation compared with the original unrestricted tree.

The final model was saved using `joblib` so that it can be reused without retraining.

Example:

```python
import joblib

joblib.dump(tree_model, "hr_attrition_decision_tree.pkl")
```

The model can later be loaded using:

```python
loaded_model = joblib.load("hr_attrition_decision_tree.pkl")
```

---

## Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook / VS Code
- joblib

---

## Future Improvements

This project reflects my current stage of learning, so I intentionally built it progressively rather than introducing techniques before learning them properly.

Future improvements may include:

- Precision, recall, and F1-score analysis in more detail
- Decision Tree visualisation
- Feature importance analysis
- Cross-validation
- Hyperparameter tuning
- Random Forest classification
- Comparing additional classification algorithms
- Building a simple interface where HR users can enter employee information and receive an attrition prediction

---

## Conclusion

This project started with a simple Logistic Regression model producing around 76–77% accuracy. Initially, that appeared to be a reasonable result, but further evaluation showed that the model was identifying only around 23–24% of employees who actually left.

Exploring the data revealed several non-linear relationships, particularly around project workload and time spent at the company. Moving to a Decision Tree allowed the model to capture these patterns much more effectively.

The first Decision Tree improved test accuracy to around 96%, but its 100% training accuracy showed that it was overfitting. After reducing the model's complexity, the final version achieved approximately 97.6% training accuracy and 97.1% testing accuracy while correctly identifying around 92% of employees who left.

The project therefore became more than an exercise in increasing accuracy. It demonstrated the importance of understanding the data, selecting appropriate evaluation metrics, comparing algorithms, recognising overfitting, and focusing on how well a model performs on the business problem it is actually intended to solve.
