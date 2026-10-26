import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from safeguard.data.validation import validate_training_data
from safeguard.detection.ml_detector import MLDetector
from safeguard.paths import PROCESSED_DIR, REPORTS_DIR, ensure_runtime_dirs
import matplotlib.pyplot as plt
import seaborn as sns

def load_twibot20_data():
    """Load TwiBot-20 data"""
    with open(PROCESSED_DIR / 'twibot20_converted_simple.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['samples'], data['labels']

def extract_features_for_evaluation(data, detector=None):
    """Extract features for evaluation"""
    detector = detector if detector is not None else MLDetector()
    X = []
    for user_data in data:
        features = detector.extract_features(user_data)
        feature_vector = [features[f] for f in sorted(features.keys())]
        X.append(feature_vector)
    return np.array(X)

def comprehensive_evaluation():
    """Comprehensive evaluation function"""
    print("=== TwiBot-20 Dataset Comprehensive Evaluation ===\n")
    
    # Load data
    samples, labels = load_twibot20_data()
    samples, labels = validate_training_data(samples, labels)
    if min(labels.count(0), labels.count(1)) < 3:
        raise ValueError('评估至少需要每类 3 个样本，以划分测试集和交叉验证集')
    print(f"Dataset size: {len(samples)} samples")
    print(f"Label distribution: Human {labels.count(0)}, Bot {labels.count(1)}")
    
    # Extract features
    print("\nExtracting features...")
    detector = MLDetector()
    X = extract_features_for_evaluation(samples, detector)
    y = np.array(labels)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining model...")
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train random forest
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = rf_classifier.predict(X_test_scaled)
    y_pred_proba = rf_classifier.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    print("\n=== Detailed Evaluation Metrics ===")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"Precision: {precision:.4f} ({precision:.2%})")
    print(f"Recall: {recall:.4f} ({recall:.2%})")
    print(f"F1-Score: {f1:.4f} ({f1:.2%})")
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC: {roc_auc:.4f} ({roc_auc:.2%})")
    except:
        print("ROC-AUC: Cannot compute (possibly imbalanced data)")
    
    # Cross-validation
    print("\n=== Cross-validation Results ===")
    cv_folds = min(5, int(np.bincount(y_train).min()))
    # Fit preprocessing within each fold, instead of leaking validation statistics.
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds)
    print(f"{cv_folds}-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Human', 'Bot'], zero_division=0))
    
    # Confusion matrix
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion Matrix:")
    print("         Predicted")
    print("        Human  Bot")
    print(f"Actual Human  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"      Bot     {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # More metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nSpecificity: {specificity:.4f} ({specificity:.2%})")
    print(f"Sensitivity: {sensitivity:.4f} ({sensitivity:.2%})")
    
    # Feature importance
    print("\n=== Feature Importance (Top 10) ===")
    feature_names = sorted(detector.extract_features(samples[0]).keys())
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(indices))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Save results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': cm.tolist(),
        'feature_importances': dict(zip(feature_names, importances.tolist()))
    }
    
    with open(REPORTS_DIR / 'evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation results saved to evaluation_results.json")
    
    return results

def plot_results(results):
    """Plot evaluation results in English"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Main metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision'], 
              results['recall'], results['f1_score']]
    
    axes[0, 0].bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    axes[0, 0].set_title('Main Evaluation Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(values):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # 2. Confusion matrix heatmap
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Human', 'Bot'], 
                yticklabels=['Human', 'Bot'], ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted Label')
    axes[0, 1].set_ylabel('True Label')
    
    # 3. Feature importance
    importances = results['feature_importances']
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    feature_names = [x[0] for x in top_features]
    importance_values = [x[1] for x in top_features]
    
    axes[1, 0].barh(range(len(feature_names)), importance_values)
    axes[1, 0].set_yticks(range(len(feature_names)))
    axes[1, 0].set_yticklabels(feature_names)
    axes[1, 0].set_title('Feature Importance (Top 10)')
    axes[1, 0].set_xlabel('Importance')
    
    # 4. Precision distribution
    axes[1, 1].pie([results['precision'], 1-results['precision']], 
                   labels=['Precision', '1-Precision'], autopct='%1.1f%%')
    axes[1, 1].set_title('Precision Distribution')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Evaluation chart saved to evaluation_results.png")

if __name__ == "__main__":
    ensure_runtime_dirs()
    # Run comprehensive evaluation
    results = comprehensive_evaluation()
    
    # Plot results
    try:
        plot_results(results)
    except Exception as e:
        print(f"Plotting error: {e}")
        print("Please make sure matplotlib and seaborn are installed: pip install matplotlib seaborn")
    
    print("\n=== Evaluation Summary ===")
    print(f"Model performance on TwiBot-20 dataset:")
    print(f"- F1-Score: {results['f1_score']:.2%}")
    print(f"- Precision: {results['precision']:.2%}")
    print(f"- Recall: {results['recall']:.2%}")
    print(f"- Accuracy: {results['accuracy']:.2%}")
    print(f"- Cross-validation Accuracy: {results['cv_mean']:.2%} (±{results['cv_std']*2:.2%})") 
