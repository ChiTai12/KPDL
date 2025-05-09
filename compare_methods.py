import pandas as pd
import re
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

def old_clean_ingredients(ingredients):
    """Phương pháp làm sạch cũ"""
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

def new_clean_ingredients(ingredients):
    """Phương pháp làm sạch mới"""
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

def compare_cleaning_methods(df, recipe_name):
    """So sánh kết quả của hai phương pháp làm sạch"""
    recipe = df[df['recipe_name'] == recipe_name].iloc[0]
    
    old_cleaned = old_clean_ingredients(recipe['ingredients'])
    new_cleaned = new_clean_ingredients(recipe['ingredients'])
    
    print(f"Công thức: {recipe_name}")
    print(f"Nguyên liệu gốc: {recipe['ingredients'][:100]}...")
    print(f"Phương pháp cũ: {old_cleaned}")
    print(f"Phương pháp mới: {new_cleaned}")
    print(f"Số nguyên liệu (cũ): {len(old_cleaned)}")
    print(f"Số nguyên liệu (mới): {len(new_cleaned)}")
    print("="*50)

def compare_search_results(df, ingredients_list):
    """So sánh kết quả tìm kiếm của hai phương pháp"""
    # Chuẩn bị dữ liệu cho phương pháp cũ
    df_old = df.copy()
    df_old['cleaned_ingredients'] = df_old['ingredients'].apply(old_clean_ingredients)
    
    # Chuẩn bị dữ liệu cho phương pháp mới
    df_new = df.copy()
    df_new['cleaned_ingredients'] = df_new['ingredients'].apply(new_clean_ingredients)
    
    # Tìm kiếm với phương pháp cũ
    ingredients_set = set(ingredients_list)
    df_old['match_score'] = df_old['cleaned_ingredients'].apply(
        lambda x: len(set(x) & ingredients_set) / len(ingredients_set) if len(ingredients_set) > 0 else 0
    )
    old_results = df_old.nlargest(5, ['match_score', 'rating'])
    
    # Tìm kiếm với phương pháp mới
    df_new['match_score'] = df_new['cleaned_ingredients'].apply(
        lambda x: len(set(x) & ingredients_set) / len(ingredients_set) if len(ingredients_set) > 0 else 0
    )
    df_new['matching_count'] = df_new['cleaned_ingredients'].apply(
        lambda x: len(set(x) & ingredients_set)
    )
    new_results = df_new.nlargest(5, ['matching_count', 'match_score', 'rating'])
    
    print(f"\nTìm kiếm với nguyên liệu: {', '.join(ingredients_list)}")
    
    print("\nKết quả với phương pháp cũ:")
    for i, (_, recipe) in enumerate(old_results.iterrows(), 1):
        print(f"{i}. {recipe['recipe_name']}")
        print(f"   Điểm khớp: {recipe['match_score']:.2f}")
        print(f"   Nguyên liệu khớp: {set(recipe['cleaned_ingredients']) & ingredients_set}")
    
    print("\nKết quả với phương pháp mới:")
    for i, (_, recipe) in enumerate(new_results.iterrows(), 1):
        print(f"{i}. {recipe['recipe_name']}")
        print(f"   Số nguyên liệu khớp: {recipe['matching_count']}/{len(ingredients_list)}")
        print(f"   Điểm khớp: {recipe['match_score']:.2f}")
        print(f"   Nguyên liệu khớp: {set(recipe['cleaned_ingredients']) & ingredients_set}")
    
    print("="*50)

def main():
    # Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    df = pd.read_csv('recipes.csv/recipes.csv')
    print(f"Đã đọc {len(df)} công thức")
    
    # Loại bỏ các công thức trùng lặp dựa trên tên
    df = df.drop_duplicates(subset=['recipe_name'])
    print(f"Còn {len(df)} công thức sau khi loại bỏ trùng lặp")
    
    # So sánh phương pháp làm sạch
    print("\nSo sánh phương pháp làm sạch:")
    compare_cleaning_methods(df, "Apple Crisp")
    compare_cleaning_methods(df, "Mulligatawny Soup")
    compare_cleaning_methods(df, "Joy's Easy Banana Bread")
    compare_cleaning_methods(df, "Moroccan Beef and Lentil Stew")
    
    # So sánh kết quả tìm kiếm
    print("\nSo sánh kết quả tìm kiếm:")
    test_cases = [
        ["apple", "cinnamon", "sugar"],
        ["chicken", "rice", "carrot"],
        ["flour", "butter", "sugar", "egg"],
        ["beef", "onion", "garlic"]
    ]
    
    for ingredients in test_cases:
        compare_search_results(df, ingredients)

if __name__ == "__main__":
    main()
