# --------------------------------------------------------
# Fed-StackFPAD: Federated Learning for Face Presentation Attack Detection with Stacking to Tackle Data Heterogeneity
#
# Author: Mehmet Fatih Gündoğar
#
# Description:
# This file is part of the Fed-StackFPAD framework, a federated learning pipeline  designed for robust face presentation
# attack detection (FPAD). It combines self-supervised pretrained ViT encoders with local fine-tuning and a final
# stacking-based ensemble classifier.
#
# --------------------------------------------------------

import pickle
import numpy as np
import argparse
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
from stacked_model_type import StackedModelType


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer


def main():
    parser = argparse.ArgumentParser(
        description="Train a stacked model for FPAD ensemble."
    )
    parser.add_argument("--target", type=str, required=True, help="Target dataset name (e.g., O)")
    parser.add_argument("--stacked_model_type", type=str, required=True,
                        choices=[m.name for m in StackedModelType],
                        help="Model type: Svm, LassoRegression, RandomForest, XGBoost, LogisticRegression")
    parser.add_argument("--train_file", type=str, required=True, help="Path to validation data pickle")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test data pickle")
    parser.add_argument("--output_roc_file", type=str, required=True, help="Path to save the data for roc curve")
    parser.add_argument("--output_model_file", type=str, required=True, help="Path to save the trained model pickle")

    args = parser.parse_args()

    # Load validation data
    with open(args.train_file, "rb") as file:
        validation_data = pickle.load(file)
    X_validation = validation_data["features"][:, :-1]
    y_validation = validation_data["features"][:, -1]

    model = None

    if args.stacked_model_type == "Svm":
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        model = GridSearchCV(SVC(probability=True), param_grid, verbose=2)
        model.fit(X_validation, y_validation)
        positive_class_probabilities = model.predict_proba(X_validation)[:, 1]

    elif args.stacked_model_type == "LassoRegression":
        model = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5, random_state=10)
        model.fit(X_validation, y_validation)
        positive_class_probabilities = model.predict(X_validation)

    elif args.stacked_model_type == "LogisticRegression":
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
        logreg = LogisticRegression(max_iter=500)
        grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_validation, y_validation)
        model = grid_search.best_estimator_
        positive_class_probabilities = model.predict_proba(X_validation)[:, 1]

    elif args.stacked_model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=40, random_state=13, criterion='gini')
        model.fit(X_validation, y_validation)
        positive_class_probabilities = model.predict_proba(X_validation)[:, 1]

    elif args.stacked_model_type == "XGBoost":
        import xgboost as xgb
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc')
        model.fit(X_validation, y_validation)
        positive_class_probabilities = model.predict_proba(X_validation)[:, 1]

    else:
        raise ValueError("Unsupported stacked model type")

    # Compute metrics
    fpr, tpr, thresholds = roc_curve(y_validation, positive_class_probabilities)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    validation_eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    hter = (fpr[np.nanargmin(np.absolute(fnr - fpr))] + fnr[np.nanargmin(np.absolute(fnr - fpr))]) / 2
    auc = roc_auc_score(y_validation, positive_class_probabilities)
    balanced_acc = balanced_accuracy_score(y_validation, (positive_class_probabilities > validation_eer_threshold).astype(int))

    print("Validation fold metrics:")
    print(f"eer: {eer}")
    print(f"hter: {hter}")
    print(f"auc: {auc}")
    print(f"balanced_acc: {balanced_acc}")

    # Load test data
    with open(args.test_file, "rb") as file:
        test_data = pickle.load(file)
    X_test = test_data["features"][:, :-1]
    y_test = test_data["features"][:, -1]

    if args.stacked_model_type == "LassoRegression":
        positive_class_probabilities = model.predict(X_test)
    else:
        positive_class_probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (positive_class_probabilities > validation_eer_threshold).astype(int)
    fpr_test = np.sum((predictions == 1) & (y_test == 0)) / np.sum(y_test == 0)
    fnr_test = np.sum((predictions == 0) & (y_test == 1)) / np.sum(y_test == 1)
    hter_test = (fpr_test + fnr_test) / 2
    auc_test = roc_auc_score(y_test, predictions)
    balanced_acc_test = balanced_accuracy_score(y_test, predictions)

    print("Test fold metrics:")
    print(f"eer (validation threshold): {eer}")
    print(f"hter with threshold: {hter_test}")
    print(f"auc: {auc_test}")
    print(f"balanced_acc: {balanced_acc_test}")

    # Save data for roc curve
    fpr_values, tpr_values, _ = roc_curve(y_test, predictions)
    fnr_values = 1 - tpr_values
    np.savez(args.output_roc_file, fpr=fpr_values, fnr=fnr_values)
    print(f"Roc data saved to {args.output_roc_file}")

    # Save model
    joblib.dump(model, args.output_model_file)
    print(f"Model saved to {args.output_model_file}")


if __name__ == "__main__":
    main()
