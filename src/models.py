import numpy as np

class LogisticRegression:
    """
    Cài đặt mô hình Logistic Regression từ đầu sử dụng NumPy
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Tốc độ học cho gradient descent
    max_iter : int, default=1000
        Số lượng iterations tối đa
    tolerance : float, default=1e-6
        Tolerance để dừng training khi cost function hội tụ
    fit_intercept : bool, default=True
        Có thêm bias term hay không
    verbose : bool, default=False
        In thông tin training process
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6, 
                 fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        
        # Thuộc tính được khởi tạo sau khi fit
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.n_features = None
        self.n_samples = None
        
    def _add_intercept(self, X):
        """Thêm bias column vào feature matrix"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _sigmoid(self, z):
        """
        Hàm sigmoid với clipping để tránh overflow
        """
        z = np.clip(z, -500, 500)
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def _loss_function(self, h, y):
        """
        Tính loss function (cross-entropy loss)
        """
        # Thêm small epsilon để tránh log(0)
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        
        cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        return cost
    
    def _gradient(self, X, h, y):
        """
        Tính gradient của loss function
        """
        error = h - y
        return np.einsum('ij,i->j', X, error) / y.size
    
    def fit(self, X, y):
        """
        Train mô hình Logistic Regression sử dụng gradient descent
        """
        # Validate inputs
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim != 2:
            raise ValueError("X phải là 2D array")
        if y.ndim != 1:
            raise ValueError("y phải là 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Số samples trong X và y phải giống nhau")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y chỉ chứa values 0 và 1")
        
        self.n_samples, self.n_features = X.shape
        
        # Thêm intercept nếu cần
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        # Khởi tạo weights (Xavier/Glorot initialization)
        n_weights = X.shape[1]
        self.weights = np.random.normal(0, np.sqrt(2.0 / n_weights), n_weights)
        
        self.loss_history = []
        
        if self.verbose:
            print(f"BẮT ĐẦU TRAINING LOGISTIC REGRESSION")
            print(f"   Features: {self.n_features}")
            print(f"   Samples: {self.n_samples}")
            print(f"   Learning rate: {self.learning_rate}")
            print(f"   Max iterations: {self.max_iter}")
            print("-" * 50)
        
        # Gradient descent
        for i in range(self.max_iter):
            # Forward pass
            z = X.dot(self.weights)
            h = self._sigmoid(z)

            # Tính loss
            loss = self._loss_function(h, y)
            self.loss_history.append(loss)

            # Backward pass
            gradient = self._gradient(X, h, y)
            
            # Update weights
            self.weights -= self.learning_rate * gradient
            
            # Kiểm tra convergence
            if i > 0:
                loss_diff = abs(self.loss_history[-2] - self.loss_history[-1])
                if loss_diff < self.tolerance:
                    if self.verbose:
                        print(f" Converged tại iteration {i+1}")
                        print(f"   Loss difference: {loss_diff:.8f}")
                    break
            
            # In progress
            if self.verbose and (i + 1) % 100 == 0:
                accuracy = self._calculate_accuracy(X, y)
                print(f"   Iteration {i+1:4d}: Loss = {loss:.6f}, Accuracy = {accuracy:.3f}")
        
        if self.verbose:
            final_accuracy = self._calculate_accuracy(X, y)
            print(f"\n TRAINING HOÀN TẤT:")
            print(f"   Final loss: {self.loss_history[-1]:.6f}")
            print(f"   Final accuracy: {final_accuracy:.3f}")
            print(f"   Total iterations: {len(self.loss_history)}")

        return self
    
    def _calculate_accuracy(self, X, y):
        """Tính accuracy cho việc monitoring"""
        predictions = self.predict(X, _internal=True)
        return (predictions == y).mean()
    
    def predict_proba(self, X):
        """
        Dự đoán xác suất cho class positive
        """
        if self.weights is None:
            raise ValueError("Model chưa được train. Hãy gọi fit() trước.")
        
        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("X phải là 2D array")
        if X.shape[1] != self.n_features:
            raise ValueError(f"X phải có {self.n_features} features")
        
        # Thêm intercept nếu cần
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        # Tính xác suất
        z = X.dot(self.weights)
        prob_positive = self._sigmoid(z)
        prob_negative = 1 - prob_positive
        
        # Return probabilities cho cả 2 classes
        return np.column_stack((prob_negative, prob_positive))
    
    def predict(self, X, _internal=False):
        """
        Dự đoán class labels
        """
        if not _internal:
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            # Internal use, X đã có intercept
            z = X.dot(self.weights)
            h = self._sigmoid(z)
            return (h >= 0.5).astype(int)
    
    def get_feature_importance(self, feature_names=None):
        """
        Lấy feature importance (absolute weights)
        """
        if self.weights is None:
            raise ValueError("Model chưa được train")
        
        # Lấy weights (bỏ bias nếu có)
        if self.fit_intercept:
            feature_weights = self.weights[1:]  # Bỏ bias term
        else:
            feature_weights = self.weights
        
        importance = np.abs(feature_weights)
        
        if feature_names is not None:
            if len(feature_names) != len(importance):
                raise ValueError("Số lượng feature_names phải bằng số features")
            return dict(zip(feature_names, importance))
        
        return importance
    
    def get_model_summary(self):
        """
        Tóm tắt thông tin model
        """
        if self.weights is None:
            return {"status": "Model chưa được train"}
        
        summary = {
            "n_features": self.n_features,
            "n_samples": self.n_samples,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "actual_iter": len(self.loss_history),
            "final_loss": self.loss_history[-1],
            "fit_intercept": self.fit_intercept,
            "converged": len(self.loss_history) < self.max_iter
        }
        
        if self.fit_intercept:
            summary["bias"] = self.weights[0]
            summary["n_weights"] = len(self.weights) - 1
        else:
            summary["n_weights"] = len(self.weights)
        
        return summary


def calculate_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Tính toán các metrics cho classification
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positive
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negative
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positive
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negative
    
    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'confusion_matrix': {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        },
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity
    }
    
    # AUC-ROC nếu có probabilities
    if y_proba is not None:
        auc = calculate_auc_roc(y_true, y_proba)
        metrics['auc_roc'] = auc
    
    return metrics


def calculate_auc_roc(y_true, y_proba):
    """
    Tính AUC-ROC
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    # Sắp xếp theo probability giảm dần
    desc_score_indices = np.argsort(y_proba)[::-1]
    y_proba_sorted = y_proba[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    
    # Tính TPR và FPR tại các thresholds khác nhau using vectorized operations
    # Lấy các thresholds unique
    thresholds = np.unique(y_proba_sorted)
    thresholds = np.concatenate([[1.0], thresholds, [0.0]])
    
    # Vectorized computation của confusion matrix cho tất cả thresholds
    # Broadcasting: (n_thresholds, 1) >= (1, n_samples)
    y_pred_matrix = (y_proba[None, :] >= thresholds[:, None]).astype(int)
    
    # Vectorized calculation của TP, FP using einsum for efficiency
    tp = np.einsum('ij,j->i', y_pred_matrix, y_true)
    fp = np.einsum('ij,j->i', y_pred_matrix, 1 - y_true)
    
    # Total positives and negatives
    total_pos = np.sum(y_true)
    total_neg = len(y_true) - total_pos
    
    fn = total_pos - tp
    tn = total_neg - fp
    
    # Vectorized TPR and FPR calculation with safe division
    tprs = np.where(total_pos > 0, tp / total_pos, 0)
    fprs = np.where(total_neg > 0, fp / total_neg, 0)
    
    # Sắp xếp theo FPR
    sorted_indices = np.argsort(fprs)
    fprs = fprs[sorted_indices]
    tprs = tprs[sorted_indices]
    
    # Tính diện tích dưới curve using vectorized operations
    # Vectorized trapezoidal rule
    dx = np.diff(fprs)
    y_avg = (tprs[1:] + tprs[:-1]) / 2
    auc = np.sum(dx * y_avg)
    
    return float(np.clip(auc, 0, 1))


def print_classification_report(metrics, class_names=None):
    """
    In classification report đẹp mắt
    """
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    cm = metrics['confusion_matrix']
    
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    
    print("\nCONFUSION MATRIX:")
    print("                 Predicted")
    print(f"                 {class_names[0]:>8} {class_names[1]:>8}")
    print(f"Actual {class_names[0]:>8}  {cm['tn']:>8d} {cm['fp']:>8d}")
    print(f"       {class_names[1]:>8}  {cm['fn']:>8d} {cm['tp']:>8d}")
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"Accuracy    : {metrics['accuracy']:.4f}")
    print(f"Precision   : {metrics['precision']:.4f}")
    print(f"Recall      : {metrics['recall']:.4f}")
    print(f"F1-Score    : {metrics['f1_score']:.4f}")
    print(f"Specificity : {metrics['specificity']:.4f}")
    
    if 'auc_roc' in metrics:
        print(f"AUC-ROC     : {metrics['auc_roc']:.4f}")


def kfold_cross_validation(X, y, model_class, model_params, k=5, random_state=42):
    """
    Thực hiện K-Fold Cross Validation 
    """
    np.random.seed(random_state)
    n_samples = len(X)
    
    # Tạo indices và shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Chia thành k folds
    fold_size = n_samples // k
    fold_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    fold_details = []
    
    print(f"Performing {k}-Fold Cross Validation...")
    print(f"Total samples: {n_samples:,}")
    print(f"Samples per fold: ~{fold_size:,}")
    print("-" * 50)
    
    for fold in range(k):
        print(f"Fold {fold + 1}/{k}:", end=" ")
        
        # Tạo train và validation indices
        start_idx = fold * fold_size
        if fold == k - 1:  # Last fold gets remaining samples
            end_idx = n_samples
        else:
            end_idx = start_idx + fold_size
            
        val_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        # Chia dữ liệu
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]
        
        # Train model trên fold này
        fold_model = model_class(**model_params)
        fold_model.fit(X_train_fold, y_train_fold)
        
        # Evaluate trên validation set
        y_val_pred = fold_model.predict(X_val_fold)
        y_val_proba = fold_model.predict_proba(X_val_fold)[:, 1]
        
        # Tính metrics
        val_metrics = calculate_classification_metrics(
            y_val_fold, y_val_pred, y_val_proba
        )
        
        # Lưu scores
        fold_scores['accuracy'].append(val_metrics['accuracy'])
        fold_scores['precision'].append(val_metrics['precision'])
        fold_scores['recall'].append(val_metrics['recall'])
        fold_scores['f1'].append(val_metrics['f1_score'])
        fold_scores['auc'].append(val_metrics['auc_roc'])
        
        # Lưu details
        fold_details.append({
            'fold': fold + 1,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'metrics': val_metrics
        })
        
        print(f"Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc_roc']:.4f}")
    
    return fold_scores, fold_details


def analyze_cv_stability(cv_scores, single_split_metrics):
    """
    Phân tích stability của cross-validation results
    """
    # Tính mean và std cho mỗi metric
    metrics_summary = {}
    for metric_name, scores in cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        metrics_summary[metric_name] = {'mean': mean_score, 'std': std_score}
    
    # Tính coefficient of variation (CV = std/mean) để đánh giá stability
    cv_stability = {}
    for metric_name, summary in metrics_summary.items():
        cv_value = summary['std'] / summary['mean'] if summary['mean'] != 0 else 0
        cv_stability[metric_name] = cv_value
    
    # Overall stability assessment
    avg_cv = np.mean(list(cv_stability.values()))
    if avg_cv < 0.05:
        stability_assessment = "EXCELLENT - Model rất stable"
    elif avg_cv < 0.1:
        stability_assessment = "GOOD - Model khá stable"  
    elif avg_cv < 0.2:
        stability_assessment = "MODERATE - Model có stability trung bình"
    else:
        stability_assessment = "POOR - Model không stable"
    
    return {
        'metrics_summary': metrics_summary,
        'cv_stability': cv_stability,
        'stability_assessment': stability_assessment,
        'avg_cv': avg_cv
    }


def print_cv_results(cv_scores, cv_details, single_split_metrics=None):
    """
    In kết quả cross-validation một cách đẹp mắt
    """
    # Phân tích stability
    if single_split_metrics:
        analysis = analyze_cv_stability(cv_scores, single_split_metrics)
        metrics_summary = analysis['metrics_summary']
        cv_stability = analysis['cv_stability']
        stability_assessment = analysis['stability_assessment']
    else:
        metrics_summary = {}
        for metric_name, scores in cv_scores.items():
            metrics_summary[metric_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
        cv_stability = {}
        stability_assessment = "N/A"
    
    print(f"\n CROSS-VALIDATION RESULTS:")
    print("=" * 50)
    
    # Tính mean và std cho mỗi metric
    for metric_name, summary in metrics_summary.items():
        mean_score = summary['mean']
        std_score = summary['std']
        scores = cv_scores[metric_name]
        
        print(f"{metric_name.upper():<10}: {mean_score:.4f} ± {std_score:.4f}")
        print(f"{'Range':<10}: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        print("-" * 25)

    # Detailed fold analysis
    print(f"\n DETAILED FOLD ANALYSIS:")
    print("-" * 60)
    print(f"{'Fold':<6} │ {'Accuracy':<10} │ {'Precision':<10} │ {'Recall':<10} │ {'F1':<10} │ {'AUC':<10}")
    print("-" * 60)

    for detail in cv_details:
        fold = detail['fold']
        m = detail['metrics']
        print(f"{fold:<6} │ {m['accuracy']:<10.4f} │ {m['precision']:<10.4f} │ {m['recall']:<10.4f} │ {m['f1_score']:<10.4f} │ {m['auc_roc']:<10.4f}")

    print("-" * 60)
    print(f"{'Mean':<6} │ {metrics_summary['accuracy']['mean']:<10.4f} │ {metrics_summary['precision']['mean']:<10.4f} │ {metrics_summary['recall']['mean']:<10.4f} │ {metrics_summary['f1']['mean']:<10.4f} │ {metrics_summary['auc']['mean']:<10.4f}")
    print(f"{'Std':<6} │ {metrics_summary['accuracy']['std']:<10.4f} │ {metrics_summary['precision']['std']:<10.4f} │ {metrics_summary['recall']['std']:<10.4f} │ {metrics_summary['f1']['std']:<10.4f} │ {metrics_summary['auc']['std']:<10.4f}")
    
    if cv_stability:
        print(f"\n STABILITY ANALYSIS:")
        print("=" * 40)
        print(f"COEFFICIENT OF VARIATION (Lower = More Stable):")
        for metric_name, cv_value in cv_stability.items():
            stability_level = "EXCELLENT" if cv_value < 0.05 else "GOOD" if cv_value < 0.1 else "MODERATE" if cv_value < 0.2 else "POOR"
            print(f"  {metric_name:<10}: {cv_value:.4f} ({stability_level})")
        
        print(f"\n Overall Stability: {stability_assessment}")
        
    if single_split_metrics:
        print(f"\n COMPARISON: Cross-Validation vs Single Split")
        print("-" * 50)
        print(f"{'Metric':<12} │ {'CV Mean':<10} │ {'CV Std':<10} │ {'Single Split':<12} │ {'Difference':<10}")
        print("-" * 50)
        
        for metric_name in metrics_summary.keys():
            cv_mean = metrics_summary[metric_name]['mean']
            cv_std = metrics_summary[metric_name]['std']
            single_value = single_split_metrics[metric_name]
            difference = abs(cv_mean - single_value)
            
            print(f"{metric_name:<12} │ {cv_mean:<10.4f} │ {cv_std:<10.4f} │ {single_value:<12.4f} │ {difference:<10.4f}")

