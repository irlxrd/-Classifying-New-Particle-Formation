"""
NPF Event Classification with Voting Ensemble
============================================

This script implements a voting classifier ensemble using three models that provide
calibrated probability estimates for New Particle Formation event prediction.

Models used:
1. Random Forest - Tree-based ensemble with built-in probability estimates
2. Logistic Regression - Linear model with well-calibrated probabilities  
3. Gradient Boosting - Boosting ensemble with probability estimates

The VotingClassifier combines predictions using soft voting (probability averaging).
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    """Load and preprocess the NPF dataset"""
    print("📁 Loading data...")
    
    # Load datasets
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Create binary target (class2) from class4
    target_mapping = {'nonevent': 0, 'Ia': 1, 'Ib': 1, 'II': 1}
    train_df['class2'] = train_df['class4'].map(target_mapping)
    
    # Check class distribution
    class_dist = train_df['class2'].value_counts().sort_index()
    print(f"\nClass distribution:")
    for class_val, count in class_dist.items():
        label = 'nonevent' if class_val == 0 else 'event'
        percentage = (count / len(train_df)) * 100
        print(f"  {class_val} ({label}): {count} ({percentage:.1f}%)")
    
    # Separate features and target
    feature_cols = [col for col in train_df.columns if col not in ['class4', 'class2']]
    
    # Handle non-numeric columns
    for col in feature_cols:
        if train_df[col].dtype == 'object':
            print(f"Dropping non-numeric column: {col}")
            feature_cols.remove(col)
    
    X_train = train_df[feature_cols]
    y_train = train_df['class2']
    X_test = test_df[feature_cols]
    
    # Handle missing values
    print(f"\nHandling missing values...")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use training medians for test
    
    # Scale features
    print("🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Final feature count: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train, feature_cols

def create_voting_classifier():
    """Create voting classifier with three calibrated models"""
    print("🤖 Creating voting classifier ensemble...")
    
    # Define base classifiers with probability estimates
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1
    )
    
    lr_classifier = LogisticRegression(
        random_state=42, 
        max_iter=1000
    )
    
    gb_classifier = GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
    
    # Create voting classifier with soft voting (uses probabilities)
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_classifier),
            ('lr', lr_classifier), 
            ('gb', gb_classifier)
        ],
        voting='soft'  # Use probability estimates for voting
    )
    
    return voting_clf, {'Random Forest': rf_classifier, 'Logistic Regression': lr_classifier, 'Gradient Boosting': gb_classifier}

def evaluate_models(models, X_train, y_train, cv_folds=5):
    """Evaluate individual models and voting classifier using cross-validation"""
    print("📊 Evaluating models with cross-validation...")
    
    results = []
    
    # Add voting classifier to models
    voting_clf, base_models = create_voting_classifier()
    all_models = {**base_models, 'Voting Classifier': voting_clf}
    
    for name, model in all_models.items():
        print(f"  Evaluating {name}...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)
        cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
        
        results.append({
            'Model': name,
            'Accuracy': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
            'ROC-AUC': f"{cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}",
            'Accuracy_Mean': cv_scores.mean(),
            'ROC_AUC_Mean': cv_roc_auc.mean()
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy_Mean', ascending=False)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS:")
    print("="*60)
    print(results_df[['Model', 'Accuracy', 'ROC-AUC']].to_string(index=False))
    
    return results_df, voting_clf

def train_and_evaluate_final_model(voting_clf, X_train, y_train):
    """Train final model and evaluate on validation set"""
    print("\n🎯 Training final voting classifier...")
    
    # Split for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train voting classifier
    voting_clf.fit(X_train_split, y_train_split)
    
    # Make predictions
    y_pred = voting_clf.predict(X_val)
    y_pred_proba = voting_clf.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\nValidation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['nonevent', 'event']))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['nonevent', 'event'],
               yticklabels=['nonevent', 'event'])
    plt.title('Confusion Matrix - Voting Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return voting_clf, accuracy, roc_auc

def generate_test_predictions(model, X_test):
    """Generate predictions for test set"""
    print("\n🔮 Generating test predictions...")
    
    # Retrain on full training set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Show prediction distribution
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    print(f"\nTest Prediction Distribution:")
    for class_val, count in pred_dist.items():
        label = 'nonevent' if class_val == 0 else 'event'
        percentage = (count / len(y_pred)) * 100
        print(f"  {class_val} ({label}): {count} ({percentage:.1f}%)")
    
    return y_pred, y_pred_proba

def save_submission(predictions, probabilities, filename='submission.csv'):
    """Save predictions to submission file with required format"""
    # For the primary binary task, we'll predict class2 but submit as simplified class4
    # Since the task is binary classification, we'll use 'nonevent' for 0 and 'event' for 1
    # But to match sample format, let's use 'Ia' as the default event type
    class4_predictions = ['nonevent' if pred == 0 else 'Ia' for pred in predictions]
    
    # Read test.csv to get proper ID starting point
    try:
        test_df = pd.read_csv('test.csv')
        start_id = len(pd.read_csv('train.csv'))  # IDs continue from training data
    except:
        start_id = 0
    
    submission_df = pd.DataFrame({
        'id': range(start_id, start_id + len(predictions)),
        'class4': class4_predictions,
        'p': probabilities
    })
    submission_df.to_csv(filename, index=False)
    print(f"\n💾 Submission saved to: {filename}")
    print(f"Columns: {list(submission_df.columns)}")
    print(f"ID range: {start_id} to {start_id + len(predictions) - 1}")
    print(f"Sample rows:")
    print(submission_df.head())
    return submission_df

def main():
    """Main execution function"""
    print("🚀 Starting NPF Event Classification Pipeline")
    print("="*60)
    
    # Load and preprocess data
    X_train, X_test, y_train, feature_names = load_and_preprocess_data()
    
    # Create and evaluate models
    results_df, voting_clf = evaluate_models({}, X_train, y_train)
    
    # Train final model
    final_model, accuracy, roc_auc = train_and_evaluate_final_model(voting_clf, X_train, y_train)
    
    # Retrain on full dataset for final predictions
    print("\n🔄 Retraining on full dataset...")
    final_model.fit(X_train, y_train)
    
    # Generate test predictions
    test_predictions, test_probabilities = generate_test_predictions(final_model, X_test)
    
    # Save submission
    submission_df = save_submission(test_predictions, test_probabilities)
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Task: Binary Classification (class2: event vs nonevent)")
    print(f"Final Model: Voting Classifier (RF + LR + GB)")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")
    print(f"Test Predictions: {len(test_predictions)} samples")
    print(f"Output File: submission.csv")
    print(f"Note: Binary predictions mapped to class4 format for submission")
    
    return final_model, submission_df

if __name__ == "__main__":
    model, submission = main()


""" NPF Classification for all 4 event classes: nonevent, 1a, 1b, and 2

The following methods use same logic as the previous task for binary classes,
event or nonevent, but implemented also to the distinct event types.

Methods used:

1. Random Forest 
2. Logistic Regression 
3. Gradient Boosting """

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    """Load and preprocess the NPF dataset"""
    print("📁 Loading data...")
    
    # Load datasets
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f'Class distribution in class4: \n {train_df['class4'].value_counts()}')
    
    # Now we have all 4 event types directly
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['class4'])
    
    # Separate features and target
    feature_cols = [col for col in train_df.columns if col not in ['class4']]
    feature_cols = [c  for c in feature_cols if train_df[c].dtype != 'object']
    
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # Handle missing values
    print(f"\nHandling missing values...")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use training medians for test
    

    print(f"Final feature count: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, feature_cols, le

def create_voting_classifier():
    """Create voting classifier with three calibrated models"""
    print("🤖 Creating voting classifier ensemble...")
    
    # Define base classifiers with probability estimates
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1
    )

    # Addded scaling here for other methods do not require it. used sklearn Pipeline
    lr_classifier = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(
        random_state=42, 
        max_iter=1000, multi_class='multinomial', solver='lbfgs')
    )])

    gb_raw  = GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
    gb_classifier = CalibratedClassifierCV(gb_raw, method = 'sigmoid', cv = 3)

    
    # Create voting classifier with soft voting (uses probabilities)
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_classifier),
            ('lr', lr_classifier), 
            ('gb', gb_classifier)
        ],
        voting='soft'  # Use probability estimates for voting
    )
    
    return voting_clf, {'Random Forest': rf_classifier, 'Logistic Regression': lr_classifier, 'Gradient Boosting': gb_classifier}

def evaluate_models(models, X_train, y_train, cv_folds=5):
    """Evaluate individual models and voting classifier using cross-validation"""
    print("📊 Evaluating models with cross-validation...")
    
    results = []
    
    # Add voting classifier to models
    voting_clf, base_models = create_voting_classifier()
    all_models = {**base_models, 'Voting Classifier': voting_clf}
    
    for name, model in all_models.items():
        print(f"  Evaluating {name}...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)
        cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='roc_auc_ovr_weighted', n_jobs=-1)
        
        results.append({
            'Model': name,
            'Accuracy': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
            'ROC-AUC': f"{cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}",
            'Accuracy_Mean': cv_scores.mean(),
            'ROC_AUC_Mean': cv_roc_auc.mean()
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy_Mean', ascending=False)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS:")
    print("="*60)
    print(results_df[['Model', 'Accuracy', 'ROC-AUC']].to_string(index=False))
    
    return results_df, voting_clf

def train_and_evaluate_final_model(voting_clf, X_train, y_train):
    """Train final model and evaluate on validation set"""
    print("\n🎯 Training final voting classifier...")
    
    # Split for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train voting classifier
    voting_clf.fit(X_train_split, y_train_split)
    
    # Make predictions
    y_pred = voting_clf.predict(X_val)
    y_pred_proba = voting_clf.predict_proba(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class = 'ovr', average = 'weighted')
    
    print(f"\nValidation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=voting_clf.classes_,
               yticklabels=voting_clf.classes_)
    plt.title('Confusion Matrix - Voting Classifier 4class')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return voting_clf, accuracy, roc_auc

def generate_test_predictions(model, X_test):
    """Generate predictions for test set"""
    print("\n🔮 Generating test predictions...")
    
    # Retrain on full training set
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Show prediction distribution
    print(f"\nTest Prediction Distribution:")
    print(pd.Series(predictions).value_counts())
    return predictions, probabilities

def save_submission(predictions, probabilities, model, filename='submission.csv'):
    """Save predictions to submission file with required format"""
    class_labels = list(model.classes_)
    
    proba_df = pd.DataFrame(probabilities, columns=[f'p{c}' for c in model.classes_])

    submission = pd.DataFrame({'id': range(len(predictions)), 'class4' : predictions})
    submission = pd.concat([submission, proba_df], axis = 1)
    submission.to_csv(filename, index = False)
    
    print(f"\n💾 Submission saved to: {filename}")
    print(submission.head())
    
    return submission

def main():
    """Main execution function"""
    print("🚀 Starting NPF Event Classification Pipeline")
    print("="*60)
    
    # Load and preprocess data
    X_train, X_test, y_train, feature_names, le = load_and_preprocess_data()
    
    # Create and evaluate models
    results_df, voting_clf = evaluate_models({}, X_train, y_train)
    
    # Train final model
    final_model, accuracy, roc_auc = train_and_evaluate_final_model(voting_clf, X_train, y_train)
    
    # Retrain on full dataset for final predictions
    print("\n🔄 Retraining on full dataset...")
    final_model.fit(X_train, y_train)
    
    predictions, probabilities = generate_test_predictions(final_model, X_test)
    # Save submission
    submission = save_submission(predictions, probabilities, final_model)
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Task event classification for class4: nonevent, 1a, 1b, 2")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")
    print(f"Test Predictions: {len(predictions)} samples")
    print(f"Output File: submission.csv")
       
    return final_model, submission

if __name__ == "__main__":
    model, submission = main()