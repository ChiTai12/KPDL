import pandas as pd
import re
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

def clean_ingredients(ingredients):
    """Làm sạch danh sách nguyên liệu"""
    # Chuyển thành chữ thường
    ingredients = ingredients.lower()
    # Tách các nguyên liệu
    ingredients_list = [item.strip() for item in ingredients.split(',')]
    # Loại bỏ số lượng và đơn vị đo
    cleaned_ingredients = []
    for item in ingredients_list:
        # Loại bỏ số và đơn vị đo
        item = re.sub(r'\d+\s*(g|kg|ml|l|cup|cups|tbsp|tsp|oz|pound|pounds|inch|inches)', '', item)
        item = re.sub(r'\d+', '', item)
        # Loại bỏ ký tự đặc biệt
        item = re.sub(r'[^\w\s]', '', item)
        # Loại bỏ khoảng trắng thừa
        item = ' '.join(item.split())
        if item:
            cleaned_ingredients.append(item)
    return cleaned_ingredients

def load_and_preprocess_data():
    """Đọc và tiền xử lý dữ liệu"""
    # Đọc dữ liệu
    df = pd.read_csv('recipes.csv/recipes.csv')
    
    # Làm sạch dữ liệu
    df['cleaned_ingredients'] = df['ingredients'].apply(clean_ingredients)
    
    # Xử lý các giá trị null
    df['prep_time'] = df['prep_time'].fillna('0 mins')
    df['cook_time'] = df['cook_time'].fillna('0 mins')
    df['total_time'] = df['total_time'].fillna('0 mins')
    df['rating'] = df['rating'].fillna(0)
    
    return df

def find_ingredient_relationships(df, min_support=0.01):
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

def get_recipes_by_ingredients(df, frequent_itemsets, ingredients_list, n_recommendations=5):
    """Gợi ý công thức dựa trên danh sách nguyên liệu"""
    # Chuyển đổi danh sách nguyên liệu thành set để so sánh
    ingredients_set = set(ingredients_list)
    
    # Tính điểm cho mỗi công thức dựa trên số nguyên liệu khớp
    df['match_score'] = df['cleaned_ingredients'].apply(
        lambda x: len(set(x) & ingredients_set) / len(ingredients_set) if len(ingredients_set) > 0 else 0
    )
    
    # Đánh giá dựa trên các tập mục phổ biến
    df['fp_score'] = 0.0
    
    # Kiểm tra xem các nguyên liệu nhập vào có xuất hiện trong các tập phổ biến không
    for _, row in frequent_itemsets.iterrows():
        itemset = set(row['itemsets'])
        # Nếu tập nguyên liệu phổ biến có ít nhất một nguyên liệu khớp với nguyên liệu nhập vào
        if len(itemset & ingredients_set) > 0:
            # Tính điểm cho các công thức có chứa tập nguyên liệu phổ biến này
            df['fp_score'] = df.apply(
                lambda recipe: recipe['fp_score'] + (
                    row['support'] * len(itemset & set(recipe['cleaned_ingredients'])) / len(itemset)
                    if len(itemset & set(recipe['cleaned_ingredients'])) > 0 else 0
                ),
                axis=1
            )
    
    # Kết hợp điểm match_score và fp_score với trọng số 70%-30%
    df['final_score'] = 0.7 * df['match_score'] + 0.3 * (df['fp_score'] / df['fp_score'].max() if df['fp_score'].max() > 0 else 0)
    
    # Sắp xếp theo điểm số tổng hợp và rating
    recommended_recipes = df.nlargest(n_recommendations, ['final_score', 'rating'])
    
    return recommended_recipes

def main():
    # Đọc và tiền xử lý dữ liệu
    print("Đang đọc và xử lý dữ liệu...")
    df = load_and_preprocess_data()
    print(f"Đã đọc {len(df)} công thức")
    
    # Tìm mối quan hệ giữa các nguyên liệu
    print("Đang tìm mối quan hệ giữa các nguyên liệu...")
    frequent_itemsets, similarity_matrix = find_ingredient_relationships(df)
    print(f"Đã tìm thấy {len(frequent_itemsets)} tập nguyên liệu phổ biến")
    
    # Thử nghiệm với một số nguyên liệu
    test_ingredients = [
        ["apple", "cinnamon", "sugar"],
        ["chicken", "rice", "carrot"],
        ["flour", "butter", "sugar", "egg"],
        ["beef", "onion", "garlic"]
    ]
    
    for ingredients in test_ingredients:
        print("\n" + "="*50)
        print(f"Tìm công thức với nguyên liệu: {', '.join(ingredients)}")
        recommendations = get_recipes_by_ingredients(df, frequent_itemsets, ingredients)
        
        print("\nCác công thức được gợi ý:")
        for i, (_, recipe) in enumerate(recommendations.iterrows(), 1):
            print(f"{i}. {recipe['recipe_name']}")
            print(f"   Điểm khớp: {recipe['match_score']:.2f}")
            print(f"   Điểm tập phổ biến: {recipe['fp_score']:.2f}")
            print(f"   Điểm tổng hợp: {recipe['final_score']:.2f}")
            print(f"   Đánh giá: {recipe['rating']}")
            print(f"   Nguyên liệu: {recipe['ingredients'][:100]}...")

if __name__ == "__main__":
    main()
