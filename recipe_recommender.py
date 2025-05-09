import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không tương tác
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Tải các tài nguyên cần thiết của NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

# Tắt cảnh báo
warnings.filterwarnings('ignore')

def clean_ingredients(ingredients):
    """Làm sạch nhẹ nhàng danh sách nguyên liệu để giữ lại thông tin quan trọng"""
    # Chuyển thành chữ thường
    ingredients = ingredients.lower()

    # Danh sách các nguyên liệu cơ bản cần tìm
    basic_ingredients = [
        'apple', 'cinnamon', 'sugar', 'flour', 'butter', 'egg', 'milk',
        'chicken', 'beef', 'pork', 'rice', 'potato', 'carrot', 'onion',
        'garlic', 'salt', 'pepper', 'oil', 'water', 'lemon', 'cheese',
        'tomato', 'pasta', 'bread', 'cream', 'vanilla', 'chocolate',
        'banana', 'orange', 'strawberry', 'blueberry', 'raspberry',
        'avocado', 'spinach', 'kale', 'lettuce', 'cucumber', 'zucchini',
        'mushroom', 'bean', 'corn', 'pea', 'nut', 'almond', 'walnut',
        'honey', 'maple', 'syrup', 'vinegar', 'wine', 'soy', 'sauce',
        'mustard', 'yogurt', 'coconut', 'lime', 'lemon', 'ginger'
    ]

    # Tìm các nguyên liệu cơ bản trong văn bản
    found_ingredients = set()
    for basic in basic_ingredients:
        # Tìm nguyên liệu dưới dạng từ hoàn chỉnh
        pattern = r'\b' + basic + r'\b'
        if re.search(pattern, ingredients):
            found_ingredients.add(basic)

    return list(found_ingredients)

def preprocess_text(text):
    """Tiền xử lý văn bản"""
    # Chuyển thành chữ thường
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Loại bỏ stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def load_and_preprocess_data():
    """Đọc và tiền xử lý dữ liệu"""
    # Đọc dữ liệu
    df = pd.read_csv('recipes.csv/recipes.csv')

    # Làm sạch dữ liệu
    df['cleaned_ingredients'] = df['ingredients'].apply(clean_ingredients)
    df['processed_directions'] = df['directions'].apply(preprocess_text)

    # Xử lý các giá trị null
    df['prep_time'] = df['prep_time'].fillna('0 mins')
    df['cook_time'] = df['cook_time'].fillna('0 mins')
    df['total_time'] = df['total_time'].fillna('0 mins')
    df['rating'] = df['rating'].fillna(0)

    return df

def find_ingredient_relationships(df, min_support=0.005):
    """Tìm mối quan hệ giữa các nguyên liệu sử dụng FP-Growth"""
    # Chuyển đổi danh sách nguyên liệu thành ma trận nhị phân
    te = TransactionEncoder()
    te_ary = te.fit(df['cleaned_ingredients']).transform(df['cleaned_ingredients'])
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Áp dụng thuật toán FP-Growth
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

    # Tạo ma trận độ tương đồng dựa trên các tập phổ biến
    n_recipes = len(df)
    similarity_matrix = np.zeros((n_recipes, n_recipes))

    # Tính độ tương đồng giữa các công thức dựa trên số lượng nguyên liệu chung
    for i in range(n_recipes):
        for j in range(i+1, n_recipes):
            # Lấy danh sách nguyên liệu của hai công thức
            ingredients_i = set(df['cleaned_ingredients'].iloc[i])
            ingredients_j = set(df['cleaned_ingredients'].iloc[j])

            # Tính số lượng nguyên liệu chung
            common_ingredients = len(ingredients_i.intersection(ingredients_j))

            # Tính độ tương đồng dựa trên số lượng nguyên liệu chung
            # và tổng số nguyên liệu của cả hai công thức
            total_ingredients = len(ingredients_i.union(ingredients_j))
            if total_ingredients > 0:
                similarity = common_ingredients / total_ingredients
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    return frequent_itemsets, similarity_matrix

def create_recipe_recommender(df):
    """Tạo hệ thống gợi ý công thức dựa trên mối quan hệ nguyên liệu từ FP-Growth"""
    # Tìm mối quan hệ giữa các nguyên liệu và tạo ma trận độ tương đồng
    frequent_itemsets, similarity_matrix = find_ingredient_relationships(df)
    return frequent_itemsets, similarity_matrix

def get_recipe_recommendations(recipe_name, df, similarity_matrix, frequent_itemsets, n_recommendations=5):
    """Lấy các công thức được gợi ý dựa trên tên món sử dụng kết quả từ FP-Growth"""
    # Lấy index của công thức cần gợi ý
    idx = df[df['recipe_name'] == recipe_name].index[0]

    # Lấy danh sách nguyên liệu của công thức hiện tại
    current_ingredients = set(df['cleaned_ingredients'].iloc[idx])

    # Tạo một mảng numpy để lưu điểm số, nhanh hơn so với danh sách Python
    recipe_scores = np.zeros(len(df))

    # Trước tiên, lấy các điểm tương đồng từ ma trận
    for i in range(len(df)):
        if i != idx:  # Bỏ qua công thức hiện tại
            recipe_scores[i] = similarity_matrix[idx, i]

    # Điểm từ các tập phổ biến - chỉ tính cho top 50 công thức có điểm tương đồng cao nhất
    # Điều này giúp giảm thời gian tính toán đáng kể
    top_similar_indices = np.argsort(recipe_scores)[-50:]
    top_similar_indices = top_similar_indices[top_similar_indices != idx]  # Loại bỏ chính nó nếu có

    fp_scores = np.zeros(len(df))

    # Tạo từ điển lưu trữ frequent_itemsets để truy cập nhanh hơn
    frequent_itemsets_dict = {}
    for _, row in frequent_itemsets.iterrows():
        itemset = frozenset(row['itemsets'])
        frequent_itemsets_dict[itemset] = row['support']

    # Chỉ tính điểm FP cho top 50 công thức tương tự
    for i in top_similar_indices:
        other_ingredients = set(df['cleaned_ingredients'].iloc[i])
        common_ingredients = current_ingredients.intersection(other_ingredients)

        # Chỉ kiểm tra các tập con nhỏ để tăng tốc
        for size in range(1, min(4, len(common_ingredients) + 1)):
            for combo in [frozenset(c) for c in [common_ingredients]]:
                if combo in frequent_itemsets_dict:
                    fp_scores[i] += frequent_itemsets_dict[combo]

    # Chuẩn hóa fp_scores
    max_fp_score = np.max(fp_scores) if np.max(fp_scores) > 0 else 1
    fp_scores = fp_scores / max_fp_score

    # Kết hợp điểm số với trọng số 70-30
    final_scores = 0.7 * recipe_scores + 0.3 * fp_scores

    # Đặt điểm của công thức hiện tại thành -1 để loại bỏ
    final_scores[idx] = -1

    # Lấy top n_recommendations công thức có điểm cao nhất
    top_indices = np.argsort(final_scores)[-n_recommendations:][::-1]

    return df.iloc[top_indices]

def cluster_recipes(df, n_clusters=5):
    """Phân cụm công thức để tạo danh mục tự động"""
    # Tạo TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Kết hợp thông tin về nguyên liệu và hướng dẫn
    if 'combined_features' not in df.columns:
        df['combined_features'] = df['cleaned_ingredients'].apply(lambda x: ' '.join(x)) + ' ' + df['processed_directions']

    # Tạo ma trận TF-IDF
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Áp dụng PCA để giảm số chiều
    pca = PCA(n_components=min(50, tfidf_matrix.shape[1]-1))
    pca_features = pca.fit_transform(tfidf_matrix.toarray())

    # Áp dụng thuật toán K-means để phân cụm
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(pca_features)

    # Tạo tên cho mỗi cụm
    cluster_names = {}
    for cluster_id in range(n_clusters):
        # Lấy 5 công thức đại diện cho mỗi cụm
        cluster_recipes = df[df['cluster'] == cluster_id].nlargest(5, 'rating')

        # Trích xuất nguyên liệu đặc trưng
        top_ingredients = []
        for ingredients in cluster_recipes['cleaned_ingredients']:
            top_ingredients.extend(ingredients)

        # Lấy 3 nguyên liệu phổ biến nhất
        common_ingredients = pd.Series(top_ingredients).value_counts().nlargest(3).index.tolist()

        # Đặt tên cho cụm
        cluster_names[cluster_id] = f"Món ăn với {', '.join(common_ingredients)}"

    # Thêm tên cụm vào dataframe
    df['cluster_name'] = df['cluster'].map(cluster_names)

    # Tạo biểu đồ phân cụm
    plt.figure(figsize=(10, 8))
    for cluster_id in range(n_clusters):
        cluster_data = pca_features[df['cluster'] == cluster_id]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cụm {cluster_id + 1}')

    plt.title('Phân cụm công thức nấu ăn')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.legend()

    # Chuyển đổi biểu đồ thành base64 để hiển thị trên web
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return df, cluster_names, image_base64

if __name__ == "__main__":
    # Đọc và tiền xử lý dữ liệu
    df = load_and_preprocess_data()

    # Tìm mối quan hệ giữa các nguyên liệu và tạo ma trận độ tương đồng
    frequent_itemsets, similarity_matrix = find_ingredient_relationships(df)
    print("\nCác tập nguyên liệu phổ biến:")
    print(frequent_itemsets.head())

    # Tạo hệ thống gợi ý dựa trên mối quan hệ nguyên liệu
    print("\nMa trận độ tương đồng đã được tạo với kích thước:", similarity_matrix.shape)

    # Phân cụm công thức
    df_clustered, cluster_names, _ = cluster_recipes(df)
    print("\nĐã phân cụm công thức thành các danh mục:")
    for cluster_id, name in cluster_names.items():
        recipe_count = len(df_clustered[df_clustered['cluster'] == cluster_id])
        print(f"Cụm {cluster_id + 1} ({name}): {recipe_count} công thức")