import pandas as pd
import re

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

def check_ingredients_in_text(text, ingredients):
    """Kiểm tra xem nguyên liệu có trong văn bản không"""
    text = text.lower()
    found = []
    for ingredient in ingredients:
        if ingredient.lower() in text:
            found.append(ingredient)
    return found

def main():
    # Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    df = pd.read_csv('recipes.csv/recipes.csv')
    print(f"Đã đọc {len(df)} công thức")
    
    # Làm sạch dữ liệu
    df['cleaned_ingredients'] = df['ingredients'].apply(clean_ingredients)
    
    # Loại bỏ các công thức trùng lặp dựa trên tên
    df = df.drop_duplicates(subset=['recipe_name'])
    print(f"Còn {len(df)} công thức sau khi loại bỏ trùng lặp")
    
    # Các bộ nguyên liệu cần kiểm tra
    test_ingredients_sets = [
        ["apple", "cinnamon", "sugar"],
        ["chicken", "rice", "carrot"],
        ["flour", "butter", "sugar", "egg"],
        ["beef", "onion", "garlic"]
    ]
    
    # Kiểm tra từng bộ nguyên liệu
    for ingredients in test_ingredients_sets:
        print("\n" + "="*50)
        print(f"Kiểm tra công thức chứa: {', '.join(ingredients)}")
        
        # Đếm số lượng công thức chứa từng số lượng nguyên liệu
        count_by_match = {i: 0 for i in range(len(ingredients) + 1)}
        
        # Lưu các công thức khớp nhiều nhất
        best_matches = []
        max_matches = 0
        
        for idx, row in df.iterrows():
            # Kiểm tra trong danh sách nguyên liệu đã làm sạch
            cleaned_matches = set(row['cleaned_ingredients']).intersection(ingredients)
            
            # Kiểm tra trong văn bản gốc
            text_matches = check_ingredients_in_text(row['ingredients'], ingredients)
            
            # Lấy số lượng khớp lớn nhất
            match_count = max(len(cleaned_matches), len(text_matches))
            count_by_match[match_count] += 1
            
            # Lưu các công thức khớp nhiều nhất
            if match_count > max_matches:
                max_matches = match_count
                best_matches = [(row['recipe_name'], cleaned_matches, text_matches)]
            elif match_count == max_matches and match_count > 0:
                best_matches.append((row['recipe_name'], cleaned_matches, text_matches))
        
        # Hiển thị kết quả
        print(f"Thống kê số lượng công thức theo số nguyên liệu khớp:")
        for i in range(len(ingredients) + 1):
            print(f"  - Khớp {i}/{len(ingredients)} nguyên liệu: {count_by_match[i]} công thức")
        
        print(f"\nCác công thức khớp nhiều nhất ({max_matches}/{len(ingredients)} nguyên liệu):")
        for i, (name, cleaned_matches, text_matches) in enumerate(best_matches[:5], 1):
            print(f"{i}. {name}")
            print(f"   Nguyên liệu khớp trong danh sách đã làm sạch: {', '.join(cleaned_matches)}")
            print(f"   Nguyên liệu khớp trong văn bản gốc: {', '.join(text_matches)}")

if __name__ == "__main__":
    main()
