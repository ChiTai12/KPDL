import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không tương tác
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
from recipe_recommender import load_and_preprocess_data

def analyze_ingredient_rating_correlation(df):
    """Phân tích tương quan giữa nguyên liệu và đánh giá"""
    print("Đang phân tích tương quan giữa nguyên liệu và đánh giá...")

    # Lấy danh sách tất cả các nguyên liệu phổ biến (xuất hiện trong ít nhất 1% công thức)
    all_ingredients = []
    for ingredients in df['cleaned_ingredients']:
        all_ingredients.extend(ingredients)

    ingredient_counts = pd.Series(all_ingredients).value_counts()
    popular_ingredients = ingredient_counts[ingredient_counts >= len(df) * 0.01].index.tolist()

    # Tạo DataFrame để lưu kết quả phân tích
    results = []

    # Tính đánh giá trung bình cho mỗi nguyên liệu
    overall_avg_rating = df['rating'].mean()

    for ingredient in popular_ingredients:
        # Lọc các công thức có chứa nguyên liệu
        recipes_with_ingredient = df[df['cleaned_ingredients'].apply(lambda x: ingredient in x)]

        # Lọc các công thức không có nguyên liệu
        recipes_without_ingredient = df[df['cleaned_ingredients'].apply(lambda x: ingredient not in x)]

        # Tính đánh giá trung bình
        avg_rating_with = recipes_with_ingredient['rating'].mean()
        avg_rating_without = recipes_without_ingredient['rating'].mean()

        # Tính số lượng công thức
        count_with = len(recipes_with_ingredient)
        count_without = len(recipes_without_ingredient)

        # Tính độ chênh lệch so với trung bình tổng thể
        diff_from_overall = avg_rating_with - overall_avg_rating

        # Tính độ chênh lệch giữa có và không có nguyên liệu
        diff_with_without = avg_rating_with - avg_rating_without

        # Lưu kết quả
        results.append({
            'ingredient': ingredient,
            'avg_rating_with': avg_rating_with,
            'avg_rating_without': avg_rating_without,
            'count_with': count_with,
            'count_without': count_without,
            'diff_from_overall': diff_from_overall,
            'diff_with_without': diff_with_without
        })

    # Chuyển kết quả thành DataFrame
    results_df = pd.DataFrame(results)

    # Sắp xếp theo độ chênh lệch giữa có và không có nguyên liệu
    results_df = results_df.sort_values('diff_with_without', ascending=False)

    return results_df, overall_avg_rating

def create_correlation_charts(results_df, overall_avg_rating, df=None):
    """Tạo các biểu đồ phân tích tương quan"""
    print("Đang tạo biểu đồ phân tích tương quan...")

    # Tạo biểu đồ cột cho top 10 nguyên liệu có ảnh hưởng tích cực nhất
    plt.figure(figsize=(14, 8))
    top_positive = results_df.head(10)

    # Chuẩn bị dữ liệu cho biểu đồ
    ingredients = top_positive['ingredient'].tolist()
    with_ratings = top_positive['avg_rating_with'].tolist()
    without_ratings = top_positive['avg_rating_without'].tolist()

    # Tạo vị trí cho các cột
    x = np.arange(len(ingredients))
    width = 0.35

    # Tạo biểu đồ cột trực tiếp bằng matplotlib thay vì pandas
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, with_ratings, width, label='Có nguyên liệu', color='#3498db')
    rects2 = ax.bar(x + width/2, without_ratings, width, label='Không có nguyên liệu', color='#e67e22')

    # Thêm đường tham chiếu cho đánh giá trung bình tổng thể
    ax.axhline(y=overall_avg_rating, color='r', linestyle='-', label=f'Đánh giá trung bình tổng thể: {overall_avg_rating:.2f}')

    # Thêm nhãn, tiêu đề và chú thích
    ax.set_xlabel('Nguyên liệu', fontsize=12, fontweight='bold')
    ax.set_ylabel('Đánh giá trung bình', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 nguyên liệu có ảnh hưởng tích cực nhất đến đánh giá', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ingredients, rotation=45, ha='right')
    ax.legend()

    # Thêm giá trị lên đầu mỗi cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 điểm dọc
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # Lưu biểu đồ thành base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    positive_chart = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    # Tạo biểu đồ cột cho top 10 nguyên liệu có ảnh hưởng tiêu cực nhất
    plt.figure(figsize=(14, 8))
    top_negative = results_df.tail(10).iloc[::-1]  # Đảo ngược để hiển thị từ tiêu cực nhất

    # Chuẩn bị dữ liệu cho biểu đồ
    ingredients = top_negative['ingredient'].tolist()
    with_ratings = top_negative['avg_rating_with'].tolist()
    without_ratings = top_negative['avg_rating_without'].tolist()

    # Tạo vị trí cho các cột
    x = np.arange(len(ingredients))
    width = 0.35

    # Tạo biểu đồ cột trực tiếp bằng matplotlib thay vì pandas
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, with_ratings, width, label='Có nguyên liệu', color='#3498db')
    rects2 = ax.bar(x + width/2, without_ratings, width, label='Không có nguyên liệu', color='#e67e22')

    # Thêm đường tham chiếu cho đánh giá trung bình tổng thể
    ax.axhline(y=overall_avg_rating, color='r', linestyle='-', label=f'Đánh giá trung bình tổng thể: {overall_avg_rating:.2f}')

    # Thêm nhãn, tiêu đề và chú thích
    ax.set_xlabel('Nguyên liệu', fontsize=12, fontweight='bold')
    ax.set_ylabel('Đánh giá trung bình', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 nguyên liệu có ảnh hưởng tiêu cực nhất đến đánh giá', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ingredients, rotation=45, ha='right')
    ax.legend()

    # Thêm giá trị lên đầu mỗi cột
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # Lưu biểu đồ thành base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    negative_chart = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    # Tạo biểu đồ nhiệt cho tương quan giữa các nguyên liệu phổ biến nhất
    top_ingredients = results_df.head(15)['ingredient'].tolist()

    # Tạo ma trận tương quan
    correlation_matrix = np.zeros((len(top_ingredients), len(top_ingredients)))

    # Kiểm tra xem df có được truyền vào không
    if df is None:
        # Nếu không có df, tạo ma trận tương quan đơn vị
        for i in range(len(top_ingredients)):
            correlation_matrix[i, i] = 1.0
    else:
        # Tính tương quan giữa các cặp nguyên liệu
        for i, ing1 in enumerate(top_ingredients):
            for j, ing2 in enumerate(top_ingredients):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Tính số lượng công thức có cả hai nguyên liệu
                    both = df[df['cleaned_ingredients'].apply(lambda x: ing1 in x and ing2 in x)]

                    # Tính hệ số tương quan
                    ing1_recipes = df[df['cleaned_ingredients'].apply(lambda x: ing1 in x)]
                    ing2_recipes = df[df['cleaned_ingredients'].apply(lambda x: ing2 in x)]

                    if len(ing1_recipes) > 0 and len(ing2_recipes) > 0:
                        correlation = len(both) / np.sqrt(len(ing1_recipes) * len(ing2_recipes))
                    else:
                        correlation = 0

                    correlation_matrix[i, j] = correlation

    # Vẽ biểu đồ nhiệt
    fig, ax = plt.subplots(figsize=(14, 12))

    # Tạo heatmap với seaborn
    heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=top_ingredients, yticklabels=top_ingredients, ax=ax)

    # Điều chỉnh kích thước chữ và góc xoay
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Thêm tiêu đề và nhãn
    ax.set_title('Tương quan giữa các nguyên liệu phổ biến nhất', fontsize=16, fontweight='bold')

    # Điều chỉnh kích thước chữ trong các ô
    for text in heatmap.texts:
        text.set_size(9)

    fig.tight_layout()

    # Lưu biểu đồ thành base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    correlation_chart = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return positive_chart, negative_chart, correlation_chart

if __name__ == "__main__":
    # Đọc và tiền xử lý dữ liệu
    df = load_and_preprocess_data()

    # Phân tích tương quan
    results_df, overall_avg_rating = analyze_ingredient_rating_correlation(df)

    # In kết quả
    print("\nTop 10 nguyên liệu có ảnh hưởng tích cực nhất đến đánh giá:")
    print(results_df.head(10)[['ingredient', 'avg_rating_with', 'avg_rating_without', 'diff_with_without']])

    print("\nTop 10 nguyên liệu có ảnh hưởng tiêu cực nhất đến đánh giá:")
    print(results_df.tail(10)[['ingredient', 'avg_rating_with', 'avg_rating_without', 'diff_with_without']])

    # Tạo biểu đồ
    positive_chart, negative_chart, correlation_chart = create_correlation_charts(results_df, overall_avg_rating, df)

    print("\nĐã tạo xong các biểu đồ phân tích tương quan.")
