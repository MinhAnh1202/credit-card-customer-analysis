import numpy as np
import numpy as np

def load_data(path):
    """
    Load dữ liệu từ file CSV.
    """
    try:
        data = np.genfromtxt(
            path,
            delimiter=",",
            dtype=None,       # autodetect dtype
            encoding="utf-8",
            names=True        # read header as column names
        )

        # Strip quotes from string fields only
        for field in data.dtype.names:
            if data[field].dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
                data[field] = np.char.strip(data[field].astype(str), '"')
        
        return data

    except Exception as e:
        print(f"Error loading file: {e}")
        return None



def preprocess_data(data, exclude_last_n=2):
    """
    Loại bỏ các cột cuối không cần thiết khỏi dữ liệu. (ở đây mặc định là 2 cột cuối)
    """
    cols = data.dtype.names[:-exclude_last_n]
    new_dtype = [(name, data.dtype[name]) for name in cols]
    data_new = np.zeros(data.shape, dtype=new_dtype)
    for name in cols:
        data_new[name] = data[name]
    
    return data_new


def check_duplicates(data, id_column='CLIENTNUM'):
    """
    Kiểm tra dữ liệu có bị trùng lặp không dựa trên cột ID.
    """
    ids = data[id_column]
    return len(ids) != len(set(ids))


def get_numeric_columns(data, exclude_columns=['CLIENTNUM']):
    """
    Lấy danh sách các cột số trong dữ liệu.
    """
    numeric_cols = []
    for name in data.dtype.names:
        if data.dtype[name].kind in ['i', 'f']:  # int hoặc float
            if name not in exclude_columns:
                numeric_cols.append(name)
    return numeric_cols


def get_categorical_columns(data):
    """
    Lấy danh sách các cột phân loại trong dữ liệu.
    """
    categorical_cols = []
    for name in data.dtype.names:
        if data.dtype[name].kind in ['U', 'S']:  # Unicode hoặc String
            categorical_cols.append(name)
    return categorical_cols


def calculate_numeric_stats(data, columns=None):
    """
    Tính toán thống kê mô tả cho các cột số.
    """
    if columns is None:
        columns = get_numeric_columns(data)
    
    stats = {}
    for col in columns:
        values = data[col]
        stats[col] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values)
        }
    
    return stats


def calculate_categorical_distribution(data, columns=None):
    """
    Tính toán phân bố của các cột phân loại.
    """
    if columns is None:
        columns = get_categorical_columns(data)
    
    distributions = {}
    for col in columns:
        values = data[col]
        unique_values, counts = np.unique(values, return_counts=True)
        
        total = len(values)
        distribution = []
        for value, count in zip(unique_values, counts):
            percentage = (count / total) * 100
            distribution.append({
                'value': value,
                'count': count,
                'percentage': percentage
            })
        
        distributions[col] = {
            'unique_count': len(unique_values),
            'distribution': distribution,
            'has_missing': any(v in ['', 'Unknown', 'NA'] for v in unique_values)
        }
    
    return distributions


def split_target_features(data, target_column, exclude_columns=['CLIENTNUM']):
    """
    Tách dữ liệu thành target và features.
    """
    # Lấy target
    target = data[target_column]
    
    # Lấy danh sách cột features (loại bỏ target và các cột không cần thiết)
    all_exclude = exclude_columns + [target_column]
    feature_cols = [col for col in data.dtype.names if col not in all_exclude]
    
    # Tạo structured array cho features
    feature_dtype = [(name, data.dtype[name]) for name in feature_cols]
    features = np.zeros(data.shape, dtype=feature_dtype)
    for col in feature_cols:
        features[col] = data[col]
    
    return target, features, feature_cols


def get_target_distribution(target):
    """
    Tính phân bố của biến target.
    """
    unique_values, counts = np.unique(target, return_counts=True)
    total = len(target)
    
    distribution = []
    for value, count in zip(unique_values, counts):
        percentage = (count / total) * 100
        distribution.append({
            'value': value,
            'count': count,
            'percentage': percentage
        })
    
    return {
        'total_samples': total,
        'unique_values': len(unique_values),
        'distribution': distribution,
        'is_balanced': max(counts) / min(counts) <= 2.0  # Tỷ lệ không quá 2:1
    }


def create_binary_target(data, target_column='Attrition_Flag', positive_class=b'Attrited Customer'):
    """
    Tạo target binary cho classification.
    """
    target_binary = (data[target_column] == positive_class).astype(int)
    return target_binary


def standardize_features(X):
    """
    Chuẩn hóa features bằng Z-score normalization.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Tránh chia cho 0 nếu std = 0
    std = np.where(std == 0, 1, std)
    X_standardized = (X - mean) / std
    return X_standardized, mean, std


def one_hot_encode_categorical(data, categorical_cols):
    """
    Thực hiện One-Hot Encoding cho categorical variables.
    """
    if not categorical_cols:
        return np.empty((len(data), 0)), [], {}
    
    all_encoded = []
    all_feature_names = []
    encoding_info = {}
    
    for col in categorical_cols:
        values = data[col]
        # Lấy tên các category duy nhất
        unique_vals, inverse_indices = np.unique(values, return_inverse=True)
        unique_vals_sorted = sorted(unique_vals)
        encoding_info[col] = unique_vals_sorted
        
        # Vectorized one-hot encoding 
        n_categories = len(unique_vals_sorted)
        # Tạo mapping từ giá trị sang chỉ số
        val_to_idx = {val: idx for idx, val in enumerate(unique_vals_sorted)}
        
        # Map giá trị sang chỉ số
        indices = np.array([val_to_idx[val] for val in unique_vals])[inverse_indices]
        
        # Tạo ma trận one-hot
        one_hot_matrix = np.zeros((len(values), n_categories), dtype=int)
        one_hot_matrix[np.arange(len(values)), indices] = 1
        
        all_encoded.append(one_hot_matrix)
        
        # Tạo tên cột mới
        col_names = [f"{col}_{val.decode() if isinstance(val, bytes) else val}" 
                    for val in unique_vals_sorted]
        all_feature_names.extend(col_names)
    
    # Kết hợp tất cả các ma trận one-hot
    X_categorical = np.concatenate(all_encoded, axis=1) if all_encoded else np.empty((len(data), 0))
    
    return X_categorical, all_feature_names, encoding_info


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """
    Chia dữ liệu thành train/test với chiến lược stratified sampling.
    """
    np.random.seed(random_state)
    
    # Lấy indices cho từng class
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    
    # Shuffle indices
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)
    
    # Tính số samples cho test set từ mỗi class
    test_size_0 = int(len(class_0_indices) * test_size)
    test_size_1 = int(len(class_1_indices) * test_size)
    
    # Chia indices
    test_indices_0 = class_0_indices[:test_size_0]
    train_indices_0 = class_0_indices[test_size_0:]
    
    test_indices_1 = class_1_indices[:test_size_1]
    train_indices_1 = class_1_indices[test_size_1:]
    
    # Kết hợp indices
    train_indices = np.concatenate([train_indices_0, train_indices_1])
    test_indices = np.concatenate([test_indices_0, test_indices_1])
    
    # Shuffle lại để trộn các class
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return (X[train_indices], X[test_indices], 
            y[train_indices], y[test_indices],
            train_indices, test_indices)


def prepare_data_for_logistic_regression(data, target_column='Attrition_Flag', 
                                       exclude_columns=['CLIENTNUM', 'Attrition_Flag'],
                                       test_size=0.2, random_state=42):
    """
    Chuẩn bị dữ liệu hoàn chỉnh cho Logistic Regression.
    """
    # Tạo binary target
    y = create_binary_target(data, target_column)
    
    # Lấy feature columns
    feature_columns = [col for col in data.dtype.names if col not in exclude_columns]
    
    # Phân loại features bằng cách sử dụng các hàm có sẵn
    numeric_features = get_numeric_columns(data, exclude_columns=exclude_columns)
    categorical_features = get_categorical_columns(data)
    
    # Loại bỏ các categorical columns có trong exclude_columns
    categorical_features = [col for col in categorical_features if col not in exclude_columns]
    
    # Process numeric features
    X_numeric = np.column_stack([data[col] for col in numeric_features])
    X_numeric_scaled, numeric_means, numeric_stds = standardize_features(X_numeric)
    
    # Process categorical features
    X_categorical, categorical_feature_names, encoding_info = one_hot_encode_categorical(data, categorical_features)
    
    # Combine features
    X_final = np.column_stack([X_numeric_scaled, X_categorical])
    final_feature_names = numeric_features + categorical_feature_names
    
    # Train-test split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_stratified(
        X_final, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': final_feature_names,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'numeric_means': numeric_means,
        'numeric_stds': numeric_stds,
        'encoding_info': encoding_info,
        'train_indices': train_idx,
        'test_indices': test_idx
    }


def check_missing_values(data, numeric_cols=None, categorical_cols=None):
    """
    Kiểm tra missing values và unknown values trong dữ liệu.
    """
    if numeric_cols is None:
        numeric_cols = get_numeric_columns(data)
    
    if categorical_cols is None:
        categorical_cols = get_categorical_columns(data)
    
    total_samples = len(data)
    missing_info = {
        'total_samples': total_samples,
        'numeric_issues': [],
        'categorical_issues': [],
        'summary': {}
    }
    
    # Kiểm tra numeric columns using vectorized operations
    for col in numeric_cols:
        values = data[col]
        
        # Vectorized NaN detection
        if values.dtype.kind in ['f', 'i']:  # float or int
            nan_mask = np.isnan(values.astype(float, errors='ignore'))
            nan_count = np.sum(nan_mask)
        else:
            nan_count = 0
            
        # Vectorized None detection
        none_mask = np.array([v is None for v in values])
        none_count = np.sum(none_mask)
        
        # Vectorized empty string detection for string types
        empty_count = 0
        if values.dtype.kind in ['U', 'S', 'O']:
            # Use vectorized string operations where possible
            str_values = np.array([str(v).strip() if v is not None else '' for v in values])
            empty_mask = (str_values == '')
            empty_count = np.sum(empty_mask)
        
        total_missing = nan_count + none_count + empty_count
        missing_percentage = (total_missing / total_samples) * 100
        
        if total_missing > 0:
            missing_info['numeric_issues'].append({
                'column': col,
                'missing_count': total_missing,
                'missing_percentage': missing_percentage,
                'nan_count': nan_count,
                'none_count': none_count,
                'empty_count': empty_count
            })
    
    # Kiểm tra categorical columns using vectorized operations
    for col in categorical_cols:
        values = data[col]
        
        # Vectorized string processing
        str_values = np.array([str(v).strip() if v is not None else 'None' for v in values])
        
        # Các giá trị coi là missing/unknown
        missing_indicators = {
            'Unknown', 'unknown', 'UNKNOWN',
            'NA', 'N/A', 'na', 'n/a',
            'NULL', 'null', 'Null',
            'None', 'none', '',
            'Missing', 'missing', 'MISSING'
        }
        
        # Vectorized missing detection using isin-like operation
        missing_mask = np.isin(str_values, list(missing_indicators))
        missing_count = np.sum(missing_mask)
        missing_percentage = (missing_count / total_samples) * 100
        
        if missing_count > 0:
            missing_info['categorical_issues'].append({
                'column': col,
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'unique_values': len(np.unique(values)),
                'sample_values': list(np.unique(values)[:5])
            })
    
    # Tổng kết
    missing_info['summary'] = {
        'numeric_columns_with_issues': len(missing_info['numeric_issues']),
        'categorical_columns_with_issues': len(missing_info['categorical_issues']),
        'total_columns_checked': len(numeric_cols) + len(categorical_cols),
        'has_missing_data': len(missing_info['numeric_issues']) + len(missing_info['categorical_issues']) > 0
    }
    
    return missing_info


def print_missing_values_report(missing_info):
    """
    In báo cáo missing values.
    """
    print("BÁO CÁO MISSING VALUES")
    print("=" * 50)
    
    summary = missing_info['summary']
    print(f"Tổng số mẫu: {missing_info['total_samples']:,}")
    print(f"Tổng số cột kiểm tra: {summary['total_columns_checked']}")
    
    # Numeric issues
    if missing_info['numeric_issues']:
        print(f"\nNUMERIC COLUMNS CÓ MISSING VALUES:")
        for issue in missing_info['numeric_issues']:
            print(f" {issue['column']}: {issue['missing_count']} missing ({issue['missing_percentage']:.2f}%)")
    else:
        print(f"\nTất cả numeric columns không có missing values")
    
    # Categorical issues  
    if missing_info['categorical_issues']:
        print(f"\nCATEGORICAL COLUMNS CÓ MISSING/UNKNOWN VALUES:")
        for issue in missing_info['categorical_issues']:
            print(f"  {issue['column']}: {issue['missing_count']} missing ({issue['missing_percentage']:.2f}%)")
            print(f"      Unique values: {issue['unique_values']}")
    else:
        print(f"\nTất cả categorical columns không có missing/unknown values")
    
    # Tổng kết
    if summary['has_missing_data']:
        total_issues = summary['numeric_columns_with_issues'] + summary['categorical_columns_with_issues']
        print(f"\nTỔNG KẾT: {total_issues}/{summary['total_columns_checked']} columns có missing data")
    else:
        print(f"\nTUYỆT VỜI! Không có missing values trong dữ liệu!")


def create_engineered_features(data):
    """
    Tạo 3 features số mới từ dữ liệu hiện có (không có categorical features).
    """
    print("FEATURE ENGINEERING")
    print("=" * 50)
    print(f"Dữ liệu hiện tại: {len(data.dtype.names)} columns")
    
    # ===== 4 FEATURES SỐ =====
    print("\nTẠO 4 NUMERIC FEATURES MỚI:")
    
    # 1. Credit Utilization Efficiency
    credit_utilization_efficiency = data['Credit_Limit'] / (data['Total_Revolving_Bal'] + 1)
    
    # 2. Customer Activity Score 
    customer_activity_score = (data['Total_Trans_Ct'] * data['Total_Trans_Amt']) / (data['Months_on_book'] + 1)
    
    # 3. Average Transaction Amount
    avg_transaction_amount = data['Total_Trans_Amt'] / np.maximum(data['Total_Trans_Ct'], 1)
    
    print(f"1. Credit_Utilization_Efficiency: min={np.min(credit_utilization_efficiency):.2f}, max={np.max(credit_utilization_efficiency):.2f}, mean={np.mean(credit_utilization_efficiency):.2f}")
    print(f"2. Customer_Activity_Score: min={np.min(customer_activity_score):.2f}, max={np.max(customer_activity_score):.2f}, mean={np.mean(customer_activity_score):.2f}")  
    print(f"3. Avg_Transaction_Amount: min={np.min(avg_transaction_amount):.2f}, max={np.max(avg_transaction_amount):.2f}, mean={np.mean(avg_transaction_amount):.2f}")
    
    # Tạo structured array mới với features bổ sung
    current_dtype = list(data.dtype.descr)
    new_fields = [
        ('Credit_Utilization_Efficiency', 'f8'),
        ('Customer_Activity_Score', 'f8'),           
        ('Avg_Transaction_Amount', 'f8')               
    ]
    
    enhanced_dtype = current_dtype + new_fields
    data_enhanced = np.zeros(len(data), dtype=enhanced_dtype)
    
    # Copy dữ liệu cũ
    for field in data.dtype.names:
        data_enhanced[field] = data[field]
    
    # Thêm features mới
    data_enhanced['Credit_Utilization_Efficiency'] = credit_utilization_efficiency
    data_enhanced['Customer_Activity_Score'] = customer_activity_score
    data_enhanced['Avg_Transaction_Amount'] = avg_transaction_amount
    
    print(f"\nKẾT QUẢ:")
    print(f"Columns trước feature engineering: {len(data.dtype.names)}")
    print(f"Columns sau feature engineering: {len(data_enhanced.dtype.names)}")
    print(f"Số features mới: {len(new_fields)}")
    print(f"Feature engineering hoàn tất với 4 features số!")
    
    return data_enhanced


def handle_categorical_missing_values(data, missing_strategies):
    """
    Xử lý missing values cho categorical features.
    """
    data_handled = data.copy()
    imputation_info = {}
    
    for col, strategy_info in missing_strategies.items():
        strategy = strategy_info['strategy']
        values = data_handled[col]
        
        # Đếm missing trước khi xử lý
        missing_mask = np.array([str(v).strip() in ['Unknown', 'unknown', 'UNKNOWN'] for v in values])
        missing_count_before = np.sum(missing_mask)
        
        if strategy == 'mode':
            # Tìm mode, loại trừ Unknown
            non_missing_values = values[~missing_mask]
            unique_vals, counts = np.unique(non_missing_values, return_counts=True)
            mode_value = unique_vals[np.argmax(counts)]
            
            # Thực hiện imputation
            new_values = values.copy()
            new_values[missing_mask] = mode_value
            data_handled[col] = new_values
            
            imputation_info[col] = {
                'strategy': strategy,
                'mode_value': mode_value.decode() if isinstance(mode_value, bytes) else str(mode_value),
                'missing_count_before': missing_count_before,
                'missing_count_after': 0
            }
            
        elif strategy == 'create_category':
            # Giữ nguyên Unknown như category riêng
            imputation_info[col] = {
                'strategy': strategy,
                'missing_count_before': missing_count_before,
                'missing_count_after': missing_count_before,
                'note': 'Unknown được giữ lại như một category'
            }
    
    return data_handled, imputation_info


def check_data_quality_detailed(data, numeric_cols, categorical_cols):
    """
    Kiểm tra chất lượng dữ liệu chi tiết với phân tích patterns và khuyến nghị.
    """
    print("KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")
    print("=" * 60)
    
    total_samples = len(data)
    quality_issues = []
    
    # 1. Kiểm tra numeric features
    print("\nNUMERIC FEATURES - Missing Values:")
    print("-" * 40)
    numeric_issues = 0
    
    for col in numeric_cols:
        values = data[col]
        
        # Kiểm tra NaN values
        nan_count = np.sum(np.isnan(values.astype(float)))
        
        # Kiểm tra None/null values
        none_count = np.sum([v is None for v in values])
        
        # Kiểm tra empty string (có thể được convert thành NaN)
        empty_count = 0
        if values.dtype.kind in ['U', 'S', 'O']:  # String types
            empty_count = np.sum([str(v).strip() == '' for v in values])
        
        total_missing = nan_count + none_count + empty_count
        missing_percentage = (total_missing / total_samples) * 100
        
        if total_missing > 0:
            print(f"{col}: {total_missing} missing ({missing_percentage:.2f}%)")
            numeric_issues += 1
            quality_issues.append({
                'column': col, 
                'type': 'numeric', 
                'missing_count': total_missing,
                'percentage': missing_percentage
            })
        else:
            print(f"{col}: Không có missing values")
    
    if numeric_issues == 0:
        print("Tất cả numeric features đều không có missing values!")
    
    # 2. Kiểm tra categorical features
    print(f"\nCATEGORICAL FEATURES - Unknown/Missing Values:")
    print("-" * 50)
    categorical_issues = 0
    
    for col in categorical_cols:
        values = data[col]
        
        # Chuyển về string để dễ kiểm tra
        str_values = [str(v).strip() if v is not None else 'None' for v in values]
        
        # Các giá trị coi là missing/unknown
        missing_indicators = [
            'Unknown', 'unknown', 'UNKNOWN',
            'NA', 'N/A', 'na', 'n/a',
            'NULL', 'null', 'Null',
            'None', 'none', '',
            'Missing', 'missing', 'MISSING'
        ]
        
        missing_count = sum([v in missing_indicators for v in str_values])
        missing_percentage = (missing_count / total_samples) * 100
        
        # Lấy unique values để hiển thị
        unique_values = np.unique(values)
        
        print(f"\n{col}:")
        print(f"  Unique values: {len(unique_values)}")
        print(f"  Values: {[v.decode() if isinstance(v, bytes) else str(v) for v in unique_values[:10]]}")
        if len(unique_values) > 10:
            print(f"  ... và {len(unique_values) - 10} values khác")
        
        if missing_count > 0:
            print(f"  Missing/Unknown: {missing_count} ({missing_percentage:.2f}%)")
            categorical_issues += 1
            quality_issues.append({
                'column': col, 
                'type': 'categorical', 
                'missing_count': missing_count,
                'percentage': missing_percentage
            })
        else:
            print(f"  Không có missing/unknown values")
    
    if categorical_issues == 0:
        print("\n Tất cả categorical features đều không có missing/unknown values!")
    
    # 3. Tổng kết
    print(f"\nTỔNG KẾT CHẤT LƯỢNG DỮ LIỆU:")
    print("-" * 40)
    print(f"Tổng số samples: {total_samples:,}")
    print(f"Numeric features có vấn đề: {numeric_issues}/{len(numeric_cols)}")
    print(f"Categorical features có vấn đề: {categorical_issues}/{len(categorical_cols)}")
    print(f"Tổng features có vấn đề: {len(quality_issues)}/{len(numeric_cols) + len(categorical_cols)}")
    
    if quality_issues:
        print(f"\nCÁC VẤN ĐỀ PHÁT HIỆN:")
        for issue in quality_issues:
            print(f"- {issue['column']} ({issue['type']}): {issue['missing_count']} missing values ({issue['percentage']:.2f}%)")
    else:
        print(f"\n KHÔNG CÓ VẤN ĐỀ VỀ CHẤT LƯỢNG DỮ LIỆU!")
    
    return quality_issues



def standardize_features_with_info(X):
    """
    Chuẩn hóa features bằng Z-score với thông tin chi tiết.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Tránh chia cho 0 nếu std = 0
    std_safe = np.where(std == 0, 1, std)
    X_standardized = (X - mean) / std_safe
    
    standardization_info = {
        'means': mean,
        'stds': std,
        'features_with_zero_std': np.sum(std == 0),
        'mean_of_standardized': np.mean(X_standardized, axis=0),
        'std_of_standardized': np.std(X_standardized, axis=0)
    }
    
    return X_standardized, standardization_info


def handle_categorical_missing_values(data):
    """
    Xử lý missing values cho categorical features bằng cách thay thế bằng mode.
    """
    data_handled = data.copy()
    imputation_info = {}
    
    # Định nghĩa các giá trị được coi là missing
    missing_indicators = ['Unknown', 'unknown', 'UNKNOWN', 'NA', 'N/A', 'na', 'n/a', 
                         'NULL', 'null', 'Null', 'None', 'none', '', 'Missing', 'missing', 'MISSING']
    
    for col in get_categorical_columns(data):
        values = data_handled[col]
        
        # Đếm missing trước khi xử lý
        missing_mask = np.array([str(v).strip() in missing_indicators for v in values])
        missing_count_before = np.sum(missing_mask)

        # Tìm mode (giá trị phổ biến nhất), loại trừ missing values
        non_missing_values = values[~missing_mask]
            
        if len(non_missing_values) > 0:
            unique_vals, counts = np.unique(non_missing_values, return_counts=True)
            mode_value = unique_vals[np.argmax(counts)]
                
            # Thực hiện imputation - thay thế missing values bằng mode
            new_values = values.copy()
            new_values[missing_mask] = mode_value
            data_handled[col] = new_values
                
            # Kiểm tra sau khi imputation
            new_missing_mask = np.array([str(v).strip() in missing_indicators for v in new_values])
            missing_count_after = np.sum(new_missing_mask)
                
            imputation_info[col] = {
                'missing_count_before': missing_count_before,
                'missing_count_after': missing_count_after,
            }
                
    return data_handled, imputation_info


def one_hot_encode_with_info(data, categorical_cols):
    """
    Thực hiện One-Hot Encoding với thông tin chi tiết.
    """
    encoded_features = []
    feature_names = []
    encoding_detailed_info = {}
    
    print("ONE-HOT ENCODING CHI TIẾT:")
    print("-" * 40)
    
    for col in categorical_cols:
        values = data[col]
        unique_vals = sorted(np.unique(values))
        
        col_info = {
            'original_column': col,
            'unique_values': unique_vals,
            'n_unique': len(unique_vals),
            'features_created': []
        }
        
        print(f"\n{col}:")
        print(f"  Unique values ({len(unique_vals)}): {[v.decode() if isinstance(v, bytes) else str(v) for v in unique_vals]}")
        
        # Tạo one-hot encoding cho từng unique value
        for val in unique_vals:
            binary_col = (values == val).astype(int)
            encoded_features.append(binary_col)
            
            feature_name = f"{col}_{val.decode() if isinstance(val, bytes) else val}"
            feature_names.append(feature_name)
            
            count = np.sum(binary_col)
            percentage = (count / len(values)) * 100
            
            col_info['features_created'].append({
                'feature_name': feature_name,
                'count': count,
                'percentage': percentage
            })
        
        print(f"  Features tạo ra: {len(unique_vals)}")
        encoding_detailed_info[col] = col_info
    
    # Kết hợp tất cả features
    if encoded_features:
        X_categorical = np.column_stack(encoded_features)
    else:
        X_categorical = np.empty((len(data), 0))
    
    return X_categorical, feature_names, encoding_detailed_info
