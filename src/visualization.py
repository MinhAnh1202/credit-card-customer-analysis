import matplotlib.pyplot as plt
import numpy as np
import src.data_processing as dp


def plot_age_analysis(ages, figsize=(12, 5)):
    """
    Vẽ biểu đồ phân tích độ tuổi khách hàng.
    
    Parameters:
    -----------
    ages : numpy array
        Mảng chứa độ tuổi khách hàng
    figsize : tuple
        Kích thước figure
    """
    try:
        mean_age = np.mean(ages)
        median_age = np.median(ages)
        
        # Phân nhóm tuổi
        age_ranges = {
            '18-30': (ages >= 18) & (ages <= 30),
            '31-40': (ages >= 31) & (ages <= 40), 
            '41-50': (ages >= 41) & (ages <= 50),
            '51-60': (ages >= 51) & (ages <= 60),
            '61+': ages >= 61
        }
        
        plt.figure(figsize=figsize)
        
        # Subplot 1: Histogram độ tuổi
        plt.subplot(1, 2, 1)
        plt.hist(ages, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        plt.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Trung bình: {mean_age:.1f}')
        plt.axvline(median_age, color='orange', linestyle='--', linewidth=2, label=f'Trung vị: {median_age:.1f}')
        plt.xlabel('Độ tuổi')
        plt.ylabel('Số lượng khách hàng')
        plt.title('Phân bố độ tuổi khách hàng')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Bar chart theo nhóm tuổi
        plt.subplot(1, 2, 2)
        age_groups = list(age_ranges.keys())
        counts = [np.sum(age_ranges[group]) for group in age_groups]
        
        bars = plt.bar(age_groups, counts, color=['lightcoral', 'lightgreen', 'lightblue', 'lightyellow', 'lightpink'])
        plt.xlabel('Nhóm tuổi')
        plt.ylabel('Số lượng khách hàng')
        plt.title('Phân bố theo nhóm tuổi')
        
        # Thêm số liệu lên các cột
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    str(count), ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib chưa được cài đặt. Chỉ hiển thị kết quả dạng text.")


def plot_income_churn_analysis(data, target, figsize=(15, 6)):
    """
    Vẽ biểu đồ phân tích mối quan hệ giữa thu nhập và churn.
    
    Parameters:
    -----------
    data : numpy structured array
        Dữ liệu khách hàng
    target : numpy array
        Mảng target (Existing/Attrited Customer)
    figsize : tuple
        Kích thước figure
    """
    try:
        income_categories = np.unique(data['Income_Category'])
        attrition_by_income = {}
        total_customers = len(data)
        total_attrited = np.sum(target == "Attrited Customer")
        avg_churn_rate = total_attrited / total_customers * 100
        
        for income in income_categories:
            mask = data['Income_Category'] == income
            total_in_category = np.sum(mask)
            attrited_in_category = np.sum((data['Income_Category'] == income) & (target == "Attrited Customer"))
            
            churn_rate = attrited_in_category / total_in_category * 100 if total_in_category > 0 else 0
            
            attrition_by_income[income] = {
                'total': total_in_category,
                'attrited': attrited_in_category,
                'churn_rate': churn_rate
            }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Subplot 1: Biểu đồ cột so sánh số lượng khách hàng theo thu nhập
        incomes = list(attrition_by_income.keys())
        total_counts = [attrition_by_income[income]['total'] for income in incomes]
        attrited_counts = [attrition_by_income[income]['attrited'] for income in incomes]
        
        x_pos = np.arange(len(incomes))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, total_counts, width, label='Tổng khách hàng', color='lightblue', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, attrited_counts, width, label='Khách rời đi', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Nhóm thu nhập')
        ax1.set_ylabel('Số lượng khách hàng')
        ax1.set_title('Phân bố khách hàng theo thu nhập')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(incomes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Thêm số liệu lên các cột
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 50, f'{int(height):,}', 
                    ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 20, f'{int(height):,}', 
                    ha='center', va='bottom', fontsize=8)
        
        # Subplot 2: Biểu đồ tỷ lệ rời đi
        churn_rates = [attrition_by_income[income]['churn_rate'] for income in incomes]
        
        bars3 = ax2.bar(incomes, churn_rates, color='orange', alpha=0.7)
        ax2.axhline(y=avg_churn_rate, color='red', linestyle='--', linewidth=2, 
                   label=f'Trung bình: {avg_churn_rate:.1f}%')
        
        ax2.set_xlabel('Nhóm thu nhập')
        ax2.set_ylabel('Tỷ lệ rời đi (%)')
        ax2.set_title('Tỷ lệ rời đi theo nhóm thu nhập')
        ax2.set_xticklabels(incomes, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Thêm số liệu lên các cột
        for bar, rate in zip(bars3, churn_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2, f'{rate:.1f}%', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib chưa được cài đặt. Chỉ hiển thị kết quả dạng text.")


def plot_card_churn_analysis(data, target, figsize=(15, 6)):
    """
    Vẽ biểu đồ phân tích mối quan hệ giữa loại thẻ và churn.
    
    Parameters:
    -----------
    data : numpy structured array
        Dữ liệu khách hàng
    target : numpy array
        Mảng target (Existing/Attrited Customer)
    figsize : tuple
        Kích thước figure
    """
    try:
        card_categories = np.unique(data['Card_Category'])
        attrition_by_card = {}
        total_customers = len(data)
        total_attrited = np.sum(target == "Attrited Customer")
        avg_churn_rate = total_attrited / total_customers * 100
        
        for card in card_categories:
            mask = data['Card_Category'] == card
            total_in_category = np.sum(mask)
            attrited_in_category = np.sum((data['Card_Category'] == card) & (target == "Attrited Customer"))
            
            churn_rate = attrited_in_category / total_in_category * 100 if total_in_category > 0 else 0
            
            attrition_by_card[card] = {
                'total': total_in_category,
                'attrited': attrited_in_category,
                'churn_rate': churn_rate
            }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Subplot 1: Donut chart phân bố loại thẻ
        cards = list(attrition_by_card.keys())
        total_counts = [attrition_by_card[card]['total'] for card in cards]

        colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral']

        wedges, texts, autotexts = ax1.pie(
            total_counts,
            labels=cards,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            pctdistance=0.78,
            labeldistance=1.15,
            wedgeprops={'width': 0.35, 'edgecolor': 'white'}
        )

        # Vòng tròn trắng ở giữa (tạo donut)
        centre_circle = plt.Circle((0, 0), 0.55, fc='white')
        ax1.add_artist(centre_circle)

        ax1.set_title('Phân bố khách hàng theo loại thẻ', fontweight='bold')

        # Chỉnh chữ phần %
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        # Subplot 2: Biểu đồ cột tỷ lệ rời đi theo loại thẻ
        churn_rates = [attrition_by_card[card]['churn_rate'] for card in cards]

        # Sắp xếp theo tỷ lệ rời đi để dễ nhìn
        sorted_data = sorted(zip(cards, churn_rates), key=lambda x: x[1], reverse=True)
        sorted_cards, sorted_rates = zip(*sorted_data)
        
        bars = ax2.bar(sorted_cards, sorted_rates, color=['red' if rate > avg_churn_rate else 'green' for rate in sorted_rates], alpha=0.7)
        ax2.axhline(y=avg_churn_rate, color='blue', linestyle='--', linewidth=2, 
                   label=f'Trung bình: {avg_churn_rate:.1f}%')
        
        ax2.set_xlabel('Loại thẻ')
        ax2.set_ylabel('Tỷ lệ rời đi (%)')
        ax2.set_title('Tỷ lệ rời đi theo loại thẻ')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Thêm số liệu lên các cột
        for bar, rate in zip(bars, sorted_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{rate:.1f}%', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Highlight cột cao nhất
        highest_bar = bars[0]  # Đã sắp xếp nên cột đầu là cao nhất
        highest_bar.set_color('darkred')
        highest_bar.set_alpha(0.9)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib chưa được cài đặt. Chỉ hiển thị kết quả dạng text.")


def plot_transaction_comparison(data, target, figsize=(15, 6)):
    """
    Vẽ biểu đồ so sánh tổng tiền giao dịch giữa 2 nhóm khách hàng.
    
    Parameters:
    -----------
    data : numpy structured array
        Dữ liệu khách hàng
    target : numpy array
        Mảng target (Existing/Attrited Customer)
    figsize : tuple
        Kích thước figure
    """
    try:
        trans_amt_existing = data['Total_Trans_Amt'][target == "Existing Customer"]
        trans_amt_attrited = data['Total_Trans_Amt'][target == "Attrited Customer"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Subplot 1: Box plot
        data_to_plot = [trans_amt_existing, trans_amt_attrited]
        labels = ['Còn sử dụng', 'Đã rời đi']
        colors = ['lightblue', 'lightcoral']
        
        box_plot = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                              notch=True, showmeans=True)
        
        # Tô màu các box
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Tổng tiền giao dịch ($)')
        ax1.set_title('Box Plot: Tổng tiền giao dịch theo nhóm khách hàng')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Histogram so sánh
        bins = np.linspace(0, 20000, 50)  # Giới hạn để dễ nhìn
        
        ax2.hist(trans_amt_existing, bins=bins, alpha=0.6, label='Còn sử dụng', 
                color='blue', density=True)
        ax2.hist(trans_amt_attrited, bins=bins, alpha=0.6, label='Đã rời đi', 
                color='red', density=True)
        
        # Thêm đường trung bình
        ax2.axvline(np.mean(trans_amt_existing), color='blue', linestyle='--', 
                   linewidth=2, label=f'TB còn sử dụng: ${np.mean(trans_amt_existing):,.0f}')
        ax2.axvline(np.mean(trans_amt_attrited), color='red', linestyle='--', 
                   linewidth=2, label=f'TB đã rời đi: ${np.mean(trans_amt_attrited):,.0f}')
        
        ax2.set_xlabel('Tổng tiền giao dịch ($)')
        ax2.set_ylabel('Mật độ')
        ax2.set_title('Phân bố tổng tiền giao dịch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib chưa được cài đặt. Chỉ hiển thị kết quả dạng text.")


def plot_utilization_comparison(data, target, figsize=(15, 6)):
    """
    Vẽ biểu đồ so sánh tỷ lệ sử dụng thẻ giữa 2 nhóm khách hàng.
    
    Parameters:
    -----------
    data : numpy structured array
        Dữ liệu khách hàng
    target : numpy array
        Mảng target (Existing/Attrited Customer)
    figsize : tuple
        Kích thước figure
    """
    try:
        util_ratio_existing = data['Avg_Utilization_Ratio'][target == "Existing Customer"]
        util_ratio_attrited = data['Avg_Utilization_Ratio'][target == "Attrited Customer"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Subplot 1: Box plot
        data_to_plot = [util_ratio_existing, util_ratio_attrited]
        labels = ['Còn sử dụng', 'Đã rời đi']
        colors = ['lightgreen', 'lightcoral']
        
        box_plot = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                              notch=True, showmeans=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Tỷ lệ sử dụng thẻ')
        ax1.set_title('Box Plot: Tỷ lệ sử dụng thẻ theo nhóm')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Histogram so sánh
        bins = np.linspace(0, 1, 50)
        
        ax2.hist(util_ratio_existing, bins=bins, alpha=0.6, label='Còn sử dụng', 
                color='green', density=True)
        ax2.hist(util_ratio_attrited, bins=bins, alpha=0.6, label='Đã rời đi', 
                color='red', density=True)
        
        ax2.axvline(np.mean(util_ratio_existing), color='green', linestyle='--', 
                   linewidth=2, label=f'TB còn sử dụng: {np.mean(util_ratio_existing):.3f}')
        ax2.axvline(np.mean(util_ratio_attrited), color='red', linestyle='--', 
                   linewidth=2, label=f'TB đã rời đi: {np.mean(util_ratio_attrited):.3f}')
        
        ax2.set_xlabel('Tỷ lệ sử dụng thẻ')
        ax2.set_ylabel('Mật độ')
        ax2.set_title('Phân bố tỷ lệ sử dụng thẻ')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib chưa được cài đặt. Chỉ hiển thị kết quả dạng text.")


def plot_target_pie_chart(target, title="Phân bố Target", figsize=(8, 8)):
    """
    Vẽ biểu đồ tròn cho phân bố target.
    
    Parameters:
    -----------
    target : numpy array
        Mảng chứa giá trị target
    title : str
        Tiêu đề biểu đồ
    figsize : tuple
        Kích thước figure
    """
    # Tính toán phân bố
    unique_values, counts = np.unique(target, return_counts=True)
    percentages = (counts / len(target)) * 100
    
    # Màu sắc cho từng phần
    colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold'][:len(unique_values)]
    
    # Tạo biểu đồ tròn
    plt.figure(figsize=figsize)
    wedges, texts, autotexts = plt.pie(
        counts, 
        labels=unique_values, 
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=[0.05] * len(unique_values)  # Tách nhẹ các phần
    )
    
    # Tùy chỉnh text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.axis('equal')  # Đảm bảo hình tròn
    
    # Thêm legend với thông tin chi tiết
    legend_labels = [f'{value}: {count} mẫu ({percent:.1f}%)' 
                    for value, count, percent in zip(unique_values, counts, percentages)]
    plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(data, columns=None, figsize=(15, 12), bins=30):
    """
    Vẽ histogram cho các cột số.
    
    Parameters:
    -----------
    data : numpy structured array
        Dữ liệu đầu vào
    columns : list, optional
        Danh sách các cột cần vẽ. Nếu None, sẽ vẽ cho tất cả cột số
    figsize : tuple
        Kích thước figure
    bins : int
        Số bins cho histogram
    """
    if columns is None:
        columns = dp.get_numeric_columns(data)
    
    # Chọn tối đa 6 cột quan trọng để vẽ
    important_cols = columns[:6] if len(columns) > 6 else columns
    
    n_cols = len(important_cols)
    n_rows = (n_cols + 2) // 3  # Chia thành 3 cột
    
    plt.figure(figsize=figsize)
    
    for i, col in enumerate(important_cols, 1):
        plt.subplot(n_rows, 3, i)
        values = data[col]
        plt.hist(values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Phân bố {col}')
        plt.xlabel(col)
        plt.ylabel('Tần số')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(data, columns=None, figsize=(15, 8)):
    """
    Vẽ biểu đồ cột cho các biến phân loại.
    
    Parameters:
    -----------
    data : numpy structured array
        Dữ liệu đầu vào
    columns : list, optional
        Danh sách các cột cần vẽ. Nếu None, sẽ vẽ cho tất cả cột phân loại
    figsize : tuple
        Kích thước figure
    """
    if columns is None:
        columns = dp.get_categorical_columns(data)
    
    # Chọn tối đa 4 cột quan trọng để vẽ
    important_cols = columns[:4] if len(columns) > 4 else columns
    
    n_cols = len(important_cols)
    n_rows = (n_cols + 1) // 2  # Chia thành 2 cột
    
    plt.figure(figsize=figsize)
    
    for i, col in enumerate(important_cols, 1):
        plt.subplot(n_rows, 2, i)
        values = data[col]
        unique_values, counts = np.unique(values, return_counts=True)
        
        plt.bar(unique_values, counts, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title(f'Phân bố {col}')
        plt.xlabel(col)
        plt.ylabel('Số lượng')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_selected_distributions(data, numeric_cols=None, categorical_cols=None):
    """
    Vẽ biểu đồ cho các cột được chọn cụ thể.
    
    Parameters:
    -----------
    data : numpy structured array
        Dữ liệu đầu vào
    numeric_cols : list, optional
        Danh sách các cột số cần vẽ
    categorical_cols : list, optional
        Danh sách các cột phân loại cần vẽ
    """
    if numeric_cols:
        plot_numeric_distributions(data, columns=numeric_cols)
    
    if categorical_cols:
        plot_categorical_distributions(data, columns=categorical_cols)


def print_numeric_stats(stats_dict):
    """
    In thống kê mô tả cho các cột số một cách đẹp mắt.
    
    Parameters:
    -----------
    stats_dict : dict
        Dictionary chứa thống kê từ hàm calculate_numeric_stats
    """
    print("=" * 60)
    print("THỐNG KÊ MÔ TẢ CHO CÁC CỘT SỐ")
    print("=" * 60)
    
    for col, stats in stats_dict.items():
        print(f"\n{col}:")
        print(f"  Min: {stats['min']:.2f}")
        print(f"  Max: {stats['max']:.2f}")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Std: {stats['std']:.2f}")


def print_categorical_stats(distribution_dict):
    """
    In thống kê phân bố cho các cột phân loại một cách đẹp mắt.
    
    Parameters:
    -----------
    distribution_dict : dict
        Dictionary chứa phân bố từ hàm calculate_categorical_distribution
    """
    print("=" * 60)
    print("PHÂN BỐ GIÁ TRỊ CHO CÁC CỘT PHÂN LOẠI")
    print("=" * 60)
    
    for col, info in distribution_dict.items():
        print(f"\n{col}:")
        print(f"Số loại khác nhau: {info['unique_count']}")
        
        for item in info['distribution']:
            print(f"  {item['value']}: {item['count']} ({item['percentage']:.1f}%)")
        
        if info['has_missing']:
            print(f"Có giá trị thiếu hoặc không xác định!")