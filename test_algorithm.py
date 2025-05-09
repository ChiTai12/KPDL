import pandas as pd
import re
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

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

def load_and_preprocess_data():
    """Đọc và tiền xử lý dữ liệu"""
    # Đọc dữ liệu
    df = pd.read_csv('recipes.csv/recipes.csv')
    
    # Làm sạch dữ liệu với phương pháp mới
    df['cleaned_ingredients'] = df['ingredients'].apply(clean_ingredients)
    
    # Xử lý các giá trị null
    df['prep_time'] = df['prep_time'].fillna('0 mins')
    df['cook_time'] = df['cook_time'].fillna('0 mins')
    df['total_time'] = df['total_time'].fillna('0 mins')
    df['rating'] = df['rating'].fillna(0)
    
    # Loại bỏ các công thức trùng lặp dựa trên tên
    df = df.drop_duplicates(subset=['recipe_name'])
    
    return df

def find_ingredient_relationships(df, min_support=0.005):
    """Tìm mối quan hệ giữa các nguyên liệu sử dụng FP-Growth"""
    # Chuyển đổi danh sách nguyên liệu thành ma trận nhị phân
    te = TransactionEncoder()
    te_ary = te.fit(df['cleaned_ingredients']).transform(df['cleaned_ingredients'])
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Áp dụng thuật toán FP-Growth
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    
    return frequent_itemsets

def get_recipes_by_ingredients(df, frequent_itemsets, ingredients_list, n_recommendations=5):
    """Gợi ý công thức dựa trên danh sách nguyên liệu"""
    # Chuyển đổi danh sách nguyên liệu thành set để so sánh
    ingredients_set = set(ingredients_list)
    
    # Tính điểm cho mỗi công thức dựa trên số nguyên liệu khớp
    df['match_score'] = df['cleaned_ingredients'].apply(
        lambda x: len(set(x) & ingredients_set) / len(ingredients_set) if len(ingredients_set) > 0 else 0
    )
    
    # Tính thêm tỷ lệ nguyên liệu khớp so với tổng số nguyên liệu của công thức
    df['ingredient_coverage'] = df['cleaned_ingredients'].apply(
        lambda x: len(set(x) & ingredients_set) / len(x) if len(x) > 0 else 0
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
    
    # Tính số lượng nguyên liệu khớp
    df['matching_count'] = df['cleaned_ingredients'].apply(
        lambda x: len(set(x) & ingredients_set)
    )
    
    # Kết hợp điểm match_score, ingredient_coverage và fp_score với trọng số mới
    df['final_score'] = 0.6 * df['match_score'] + 0.2 * df['ingredient_coverage'] + 0.2 * (df['fp_score'] / df['fp_score'].max() if df['fp_score'].max() > 0 else 0)
    
    # Lọc các công thức có ít nhất 1 nguyên liệu khớp
    filtered_df = df[df['matching_count'] > 0]
    
    # Nếu không có công thức nào khớp, giảm yêu cầu
    if len(filtered_df) < n_recommendations:
        filtered_df = df
    
    # Sắp xếp theo số lượng nguyên liệu khớp, điểm tổng hợp và rating
    recommended_recipes = filtered_df.nlargest(n_recommendations, ['matching_count', 'final_score', 'rating'])
    
    return recommended_recipes

def test_algorithm():
    """Kiểm tra thuật toán tìm kiếm công thức"""
    print("Đang đọc và xử lý dữ liệu...")
    df = load_and_preprocess_data()
    print(f"Đã đọc {len(df)} công thức (sau khi loại bỏ trùng lặp)")
    
    print("Đang tìm mối quan hệ giữa các nguyên liệu...")
    frequent_itemsets = find_ingredient_relationships(df)
    print(f"Đã tìm thấy {len(frequent_itemsets)} tập nguyên liệu phổ biến")
    
    # Kiểm tra với các bộ nguyên liệu khác nhau
    test_cases = [
        ["apple", "cinnamon", "sugar"],
        ["chicken", "rice", "carrot"],
        ["flour", "butter", "sugar", "egg"],
        ["beef", "onion", "garlic"]
    ]
    
    for ingredients in test_cases:
        print("\n" + "="*50)
        print(f"Tìm công thức với nguyên liệu: {', '.join(ingredients)}")
        
        # Kiểm tra từng bước của thuật toán
        print("\nBước 1: Tính điểm khớp (match_score)")
        df_test = df.copy()
        ingredients_set = set(ingredients)
        df_test['match_score'] = df_test['cleaned_ingredients'].apply(
            lambda x: len(set(x) & ingredients_set) / len(ingredients_set) if len(ingredients_set) > 0 else 0
        )
        print(f"Điểm khớp trung bình: {df_test['match_score'].mean():.4f}")
        print(f"Số công thức có điểm khớp > 0: {len(df_test[df_test['match_score'] > 0])}")
        
        print("\nBước 2: Tính tỷ lệ nguyên liệu khớp (ingredient_coverage)")
        df_test['ingredient_coverage'] = df_test['cleaned_ingredients'].apply(
            lambda x: len(set(x) & ingredients_set) / len(x) if len(x) > 0 else 0
        )
        print(f"Tỷ lệ nguyên liệu khớp trung bình: {df_test['ingredient_coverage'].mean():.4f}")
        
        print("\nBước 3: Tính số lượng nguyên liệu khớp (matching_count)")
        df_test['matching_count'] = df_test['cleaned_ingredients'].apply(
            lambda x: len(set(x) & ingredients_set)
        )
        for i in range(len(ingredients) + 1):
            count = len(df_test[df_test['matching_count'] == i])
            print(f"Số công thức khớp {i}/{len(ingredients)} nguyên liệu: {count}")
        
        print("\nBước 4: Lọc và sắp xếp kết quả")
        filtered_df = df_test[df_test['matching_count'] > 0]
        print(f"Số công thức sau khi lọc: {len(filtered_df)}")
        
        # Tính điểm tổng hợp
        filtered_df['final_score'] = 0.6 * filtered_df['match_score'] + 0.2 * filtered_df['ingredient_coverage']
        
        # Sắp xếp kết quả
        recommended_recipes = filtered_df.nlargest(5, ['matching_count', 'final_score', 'rating'])
        
        print("\nKết quả cuối cùng:")
        for i, (_, recipe) in enumerate(recommended_recipes.iterrows(), 1):
            print(f"{i}. {recipe['recipe_name']}")
            print(f"   Số nguyên liệu khớp: {recipe['matching_count']}/{len(ingredients)}")
            print(f"   Điểm khớp: {recipe['match_score']:.2f}")
            print(f"   Tỷ lệ nguyên liệu khớp: {recipe['ingredient_coverage']:.2f}")
            print(f"   Điểm tổng hợp: {recipe['final_score']:.2f}")
            print(f"   Đánh giá: {recipe['rating']}")
            print(f"   Nguyên liệu khớp: {set(recipe['cleaned_ingredients']) & ingredients_set}")

if __name__ == "__main__":
    test_algorithm()
