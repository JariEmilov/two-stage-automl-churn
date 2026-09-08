# AutoML Workflow Context: LazyPredict → Auto-sklearn Optimization

## Overview
This document provides context for implementing a two-stage automated machine learning workflow that combines rapid model screening with sophisticated hyperparameter optimization for customer churn prediction.

## Workflow Strategy

### Stage 1: Rapid Model Screening with LazyPredict
**Purpose**: Quickly evaluate 40+ machine learning models to identify promising algorithm families for churn prediction

**Key Characteristics**:
- Execution time: < 1 minute
- No hyperparameter tuning (uses default configurations)
- Provides comprehensive performance metrics across all scikit-learn compatible models
- Identifies which model types (tree-based, linear, ensemble, etc.) work best for churn prediction

**Output**: 
- Performance comparison table with accuracy, F1-score, ROC-AUC, and training time
- Top 3-5 model candidates for further optimization

### Stage 2: Optimization with Auto-sklearn
**Purpose**: Perform robust Bayesian optimization on the promising models identified in Stage 1

**Key Characteristics**:
- Uses Bayesian optimization for efficient hyperparameter search
- Leverages meta-learning from previous datasets
- Automatic ensemble building for improved performance
- Handles preprocessing, feature engineering, and model selection simultaneously

**Output**:
- Optimized model with best hyperparameters
- Ensemble model combining multiple optimized configurations
- Cross-validated performance metrics

## Implementation Approach

### Step 1: Initial Data Preparation
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('customer_churn_dataset.csv')

# Separate features and target
X = df.drop('churned', axis=1)
y = df['churned']

# Train-test split with stratification (important due to 72% churn rate)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Step 2: LazyPredict Screening
```python
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import LabelEncoder

# Encode categorical features for LazyPredict
categorical_columns = ['contract_type', 'payment_method', 'internet_service', 
                       'tech_support', 'online_backup', 'paperless_billing']

# Create copy for encoding
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

# Label encode categorical features
for col in categorical_columns:
    le = LabelEncoder()
    X_train_encoded[col] = le.fit_transform(X_train[col])
    X_test_encoded[col] = le.transform(X_test[col])

# Initialize LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

# Fit and get results
models, predictions = clf.fit(X_train_encoded, X_test_encoded, y_train, y_test)

# Display top models
print("Top 10 Models from LazyPredict:")
print(models.head(10))

# Identify top performers
top_models = models.head(5).index.tolist()
print(f"\nTop performing model families: {top_models}")
```

### Step 3: Model Selection Logic for Churn Prediction
Based on LazyPredict results, select models for Auto-sklearn optimization:

**Expected Patterns for Churn Data**:
- Tree-based models (Random Forest, XGBoost, LightGBM) typically perform well due to non-linear relationships
- Gradient boosting excels at capturing complex customer behavior patterns
- Logistic Regression provides good baseline for interpretation
- Ensemble methods often achieve best results

**Decision Rules**:
- If tree-based models dominate (ROC-AUC > 0.80) → Focus on gradient boosting in Auto-sklearn
- If linear models competitive (ROC-AUC > 0.75) → Include linear classifiers for interpretability
- If ensemble methods top leaderboard → Enable Auto-sklearn's ensemble building
- Monitor F1-score closely due to class imbalance (72% churn rate)

### Step 4: Auto-sklearn Optimization
```python
import autosklearn.classification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Configure Auto-sklearn for churn prediction
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=600,  # 10 minutes
    per_run_time_limit=60,        # 1 minute per model
    ensemble_size=10,             # Number of models in ensemble
    memory_limit=3072,            # Memory in MB
    n_jobs=-1,                    # Use all CPU cores
    seed=42,
    resampling_strategy='cv',     # Cross-validation
    resampling_strategy_arguments={'folds': 5}
)

# Fit the model
automl.fit(X_train_encoded, y_train)

# Get predictions
y_pred = automl.predict(X_test_encoded)
y_pred_proba = automl.predict_proba(X_test_encoded)

# Evaluate with focus on churn-specific metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# Show models in ensemble
print("\nModels in Auto-sklearn Ensemble:")
print(automl.show_models())
```

### Step 5: Model Analysis and Churn Insights
```python
# Get leaderboard of all models tried
leaderboard = automl.leaderboard()
print("Auto-sklearn Model Leaderboard:")
print(leaderboard)

# Get statistics about the search
stats = automl.sprint_statistics()
print("\nSearch Statistics:")
print(stats)

# Feature importance analysis (if using tree-based model)
import matplotlib.pyplot as plt

# Get feature names
feature_names = X_train_encoded.columns.tolist()

# Note: Feature importance extraction depends on the final model type
print("\nTop features influencing churn (for interpretation):")
print("- Contract type (month-to-month vs. long-term)")
print("- Customer satisfaction score")
print("- Support tickets frequency")
print("- Account age (tenure)")
print("- Monthly charges relative to services")
```

## Best Practices for Churn Prediction

### 1. Handling Class Imbalance
- **Dataset**: 72% churn rate (7,193 churned vs. 2,807 retained)
- **Strategy**: Use stratified sampling in train-test split
- **Metrics**: Prioritize ROC-AUC and F1-score over accuracy
- **Auto-sklearn**: Consider adding class weights or SMOTE if needed

### 2. Feature Engineering Considerations
- **Contract Type**: Month-to-month contracts show higher churn (encode appropriately)
- **Customer Tenure**: Account age is inversely correlated with churn
- **Satisfaction Score**: Strong predictor; derived from support tickets
- **Service Utilization**: Number of services vs. monthly charges ratio
- **Payment Method**: Electronic payment correlates with lower churn

### 3. Categorical Feature Handling
- **For LazyPredict**: Use Label Encoding or One-Hot Encoding
- **For Auto-sklearn**: Can handle categorical features natively with proper configuration
- **Recommendation**: Start with Label Encoding for ordinal features (contract_type), One-Hot for nominal

### 4. Time Budget Allocation
- **LazyPredict**: < 1 minute (no configuration needed)
- **Auto-sklearn**: Allocate based on business needs
  - Quick iteration: 5-10 minutes
  - Standard workflow: 30-60 minutes
  - Production model: 2-4 hours

### 5. Business Metrics for Churn
- **False Negatives** (predicting retention but customer churns): Most costly - lost customer
- **False Positives** (predicting churn but customer stays): Cost of retention offer
- **Optimization**: Tune threshold based on business costs
- **Interpretation**: Model should identify actionable churn drivers

## Dataset-Specific Considerations

### Customer Churn Dataset Characteristics
- **Size**: 10,000 rows × 15 columns
- **Target**: Binary classification (churned: 0 = Retained, 1 = Churned)
- **Class Imbalance**: 72% churn rate (realistic for telecom/subscription services)
- **Features**: 
  - Numeric: customer_age, account_age_months, monthly_charges, total_charges, support_tickets, satisfaction_score, num_services, avg_monthly_usage_gb
  - Categorical: contract_type, payment_method, internet_service, tech_support, online_backup, paperless_billing
- **Underlying Trends**: 
  - Month-to-month contracts = higher churn
  - Low satisfaction + high support tickets = higher churn
  - Short account age = higher churn
  - Specific payment methods correlated with churn

### Expected Performance
- **LazyPredict**: Should identify tree-based models (Random Forest, XGBoost, LightGBM) as top performers with ROC-AUC > 0.75
- **Auto-sklearn**: Expected improvement of 3-7% in ROC-AUC through hyperparameter tuning
- **Key Features**: contract_type, satisfaction_score, support_tickets, account_age_months likely most important

### Business Application
- **Use Case**: Proactive customer retention
- **Intervention**: Target high-risk customers with retention offers
- **ROI**: Reducing churn by even 5% can significantly impact revenue
- **Monitoring**: Retrain model monthly to capture changing customer behavior

## Troubleshooting

### Common Issues

**LazyPredict Errors**:
- Categorical features not encoded → Apply Label Encoding or One-Hot Encoding first
- Memory errors with large datasets → Reduce dataset size for initial screening
- Model-specific failures → Use `ignore_warnings=True`
- Class imbalance warnings → Expected; focus on ROC-AUC and F1-score

**Auto-sklearn Issues**:
- Installation problems → Use conda: `conda install -c conda-forge auto-sklearn`
- Memory errors → Reduce `memory_limit` or `ensemble_size`
- Slow performance → Reduce `time_left_for_this_task` or limit model types
- Class imbalance handling → Add `resampling_strategy` with appropriate method

### Performance Tips
1. Start with LazyPredict to identify promising algorithms quickly
2. Use ROC-AUC as primary metric for imbalanced churn data
3. Configure Auto-sklearn to focus on top-performing algorithm families from LazyPredict
4. Enable ensemble methods for 2-5% performance boost
5. Save optimized model using pickle for deployment
6. Use probability thresholds (not 0.5) optimized for business costs

### Churn-Specific Considerations
- **Threshold Tuning**: Adjust prediction threshold based on cost of false negatives vs. false positives
- **Feature Interpretation**: Use SHAP or LIME to explain individual churn predictions
- **Temporal Validation**: Consider time-based splits if dataset has temporal component
- **Segment Analysis**: Analyze churn drivers for different customer segments

## Expected Workflow Timeline
1. **Data Loading & Preprocessing**: 30-60 seconds
2. **LazyPredict Screening**: 30-90 seconds
3. **Analysis of LazyPredict Results**: 2-3 minutes
4. **Auto-sklearn Optimization**: 10-60 minutes (configurable)
5. **Final Evaluation & Threshold Tuning**: 3-5 minutes
6. **Feature Importance & Insights**: 2-3 minutes

**Total**: 20-75 minutes depending on Auto-sklearn time budget

## Next Steps After This Workflow
1. **Threshold Optimization**: Determine optimal probability threshold based on business costs
2. **Feature Importance Analysis**: Identify top churn drivers for business action
3. **Segment-Specific Models**: Build models for high-value customer segments
4. **Retention Strategy**: Design targeted interventions for high-risk customers
5. **Model Deployment**: Deploy to production with monitoring
6. **A/B Testing**: Test retention offers on predicted high-risk customers
7. **Model Monitoring**: Track model performance drift and retrain regularly
8. **Business Integration**: Connect predictions to CRM and retention workflows

## Key Churn Metrics to Track
- **Churn Rate**: Overall percentage of customers who churned
- **Churn Risk Score**: Probability output from model (0-1)
- **Customer Lifetime Value (CLV)**: Prioritize retention for high-CLV customers
- **Retention ROI**: Cost of retention offer vs. value of retained customer
- **Model Performance**: ROC-AUC, F1-score, recall at different thresholds

## References
- LazyPredict Documentation: https://lazypredict.readthedocs.io
- Auto-sklearn Documentation: https://automl.github.io/auto-sklearn/
- Scikit-learn Model Selection: https://scikit-learn.org/stable/model_selection.html
- Handling Imbalanced Data: https://imbalanced-learn.org/
