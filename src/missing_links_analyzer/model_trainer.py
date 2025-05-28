
"""
Module for training and tuning XGBoost classification models.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RepeatedKFold, # Kept for potential future use if regression tasks are added
    RepeatedStratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV
)
from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
# No specific warning filters added for now, adhering to prompt.
# Global warnings.filterwarnings('ignore') removed.

def train_and_tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_base_params: dict,
    hyperparam_search_space: dict,
    cv_n_splits: int = 5,
    cv_n_repeats: int = 3,
    search_type: str = 'grid',
    n_iter_random: int = 100,
    scoring_metrics: dict[str, str] | str = 'roc_auc',
    refit_metric: str = 'roc_auc'
) -> tuple[XGBClassifier, dict, dict]:
    """
    Trains and tunes an XGBoost classifier model, evaluates it on a test set.

    Args:
        X_train: Training features.
        y_train: Training target variable.
        X_test: Test features.
        y_test: Test target variable.
        model_base_params: Base parameters for XGBClassifier 
                           (e.g., objective, scale_pos_weight, random_state).
        hyperparam_search_space: Hyperparameter grid for tuning.
        cv_n_splits: Number of splits for cross-validation. Defaults to 5.
        cv_n_repeats: Number of repeats for cross-validation. Defaults to 3.
        search_type: Type of search ('grid' or 'random'). Defaults to 'grid'.
        n_iter_random: Number of iterations for random search. Defaults to 100.
        scoring_metrics: Scoring metric(s) for tuning. Can be a string or dict.
                         Defaults to 'roc_auc'.
        refit_metric: Metric to use for refitting the best model. Defaults to 'roc_auc'.

    Returns:
        tuple[XGBClassifier, dict, dict]: 
            - best_model: The best trained XGBClassifier model.
            - best_hyperparams: Dictionary of the best hyperparameters found (including optimal_threshold).
            - test_evaluation_metrics: Dictionary of evaluation metrics on the test set.
    """
    logger.info("Initializing XGBClassifier model with base parameters.")
    model = XGBClassifier(**model_base_params)

    logger.info(f"Setting up RepeatedStratifiedKFold cross-validation: "
                f"n_splits={cv_n_splits}, n_repeats={cv_n_repeats}, "
                f"random_state={model_base_params.get('random_state')}.")
    cv = RepeatedStratifiedKFold(
        n_splits=cv_n_splits, 
        n_repeats=cv_n_repeats, 
        random_state=model_base_params.get('random_state')
    )
  
    logger.info(f"Defining search strategy: type='{search_type}'. Refit metric: '{refit_metric}'.")
    if search_type == 'grid':
        search = GridSearchCV(
            model, 
            hyperparam_search_space, 
            scoring=scoring_metrics, 
            n_jobs=-1, 
            cv=cv, 
            refit=refit_metric,
            verbose=0 # Using logger instead
        )
    elif search_type == 'random':
        search = RandomizedSearchCV(
            model, 
            hyperparam_search_space, 
            scoring=scoring_metrics, 
            n_jobs=-1, 
            cv=cv, 
            n_iter=n_iter_random, 
            refit=refit_metric,
            verbose=0 # Using logger instead
        )
    else:
        raise ValueError(f"Unsupported search_type: '{search_type}'. Must be 'grid' or 'random'.")
    
    logger.info(f"Executing hyperparameter search for XGBClassifier...")
    search_result = search.fit(X_train, y_train)
    
    best_model = search_result.best_estimator_
    # These are only the parameters that were searched over by GridSearchCV/RandomizedSearchCV
    best_hyperparams_tuned = search_result.best_params_ 
    
    # Combine base params with tuned params for a full picture, tuned ones take precedence
    best_hyperparams_full = {**model_base_params, **best_hyperparams_tuned}

    logger.info(f"Best CV score ({refit_metric}): {search_result.best_score_:.4f}")
    logger.info(f"Best tuned hyperparameters: {best_hyperparams_tuned}")

    logger.info("Calculating optimal classification threshold based on ROC curve (TPR-FPR).")
    y_pred_proba_train = best_model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba_train)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    logger.info(f"Optimal threshold found: {optimal_threshold:.4f}")
    
    # Add optimal_threshold to the hyperparameters dictionary to be returned
    best_hyperparams_full['optimal_threshold'] = optimal_threshold

    logger.info("Evaluating the best model on the test set using the optimal threshold.")
    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred_test_binary = (y_pred_test_proba >= optimal_threshold).astype(int)

    test_accuracy = accuracy_score(y_test, y_pred_test_binary)
    test_precision = precision_score(y_test, y_pred_test_binary, zero_division=0) # Handle zero division
    test_recall = recall_score(y_test, y_pred_test_binary, zero_division=0)    # Handle zero division
    test_f1 = f1_score(y_test, y_pred_test_binary, zero_division=0)            # Handle zero division
    test_roc_auc = roc_auc_score(y_test, y_pred_test_proba) # Use probabilities for ROC AUC
    test_conf_matrix = confusion_matrix(y_test, y_pred_test_binary)

    test_evaluation_metrics = {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'roc_auc_score': test_roc_auc,
        'confusion_matrix': test_conf_matrix.tolist(), # Convert to list for easier logging/serialization
        'optimal_threshold_used': optimal_threshold
    }

    logger.info("Test set evaluation metrics:")
    for metric, value in test_evaluation_metrics.items():
        if metric == 'confusion_matrix':
            logger.info(f"  {metric}:")
            for row in value: # Log matrix row by row for readability
                logger.info(f"    {row}")
        else:
            logger.info(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
            
    # The returned best_hyperparams should ideally be the full set used by the best_model
    # which includes base_params, tuned_params, and the derived optimal_threshold.
    return best_model, best_hyperparams_full, test_evaluation_metrics
