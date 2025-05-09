from recipe_recommender import load_and_preprocess_data, find_ingredient_relationships
import pandas as pd

def main():
    # Đọc và tiền xử lý dữ liệu
    print("Đang đọc và xử lý dữ liệu...")
    df = load_and_preprocess_data()
    print(f"Đã đọc {len(df)} công thức (sau khi loại bỏ trùng lặp)")
    
    # Tìm mối quan hệ giữa các nguyên liệu
    print("Đang tìm mối quan hệ giữa các nguyên liệu...")
    frequent_itemsets, _ = find_ingredient_relationships(df)
    print(f"Đã tìm thấy {len(frequent_itemsets)} tập nguyên liệu phổ biến")
    
    # In ra các tập nguyên liệu phổ biến có 1 phần tử
    print("\nCác nguyên liệu phổ biến (1 phần tử):")
    single_items = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 1)]
    # Sắp xếp theo độ hỗ trợ giảm dần
    single_items = single_items.sort_values('support', ascending=False)
    
    for _, row in single_items.iterrows():
        ingredient = list(row['itemsets'])[0]
        count = int(row['support'] * len(df))
        percentage = row['support'] * 100
        print(f"{ingredient}: {count} lần xuất hiện ({percentage:.2f}%)")
    
    # In ra một số tập nguyên liệu phổ biến có 2 phần tử
    print("\nMột số tập nguyên liệu phổ biến (2 phần tử):")
    pair_items = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)]
    # Sắp xếp theo độ hỗ trợ giảm dần
    pair_items = pair_items.sort_values('support', ascending=False)
    
    # In ra 20 tập đầu tiên
    for _, row in pair_items.head(20).iterrows():
        ingredients = list(row['itemsets'])
        count = int(row['support'] * len(df))
        percentage = row['support'] * 100
        print(f"{ingredients[0]} + {ingredients[1]}: {count} lần xuất hiện cùng nhau ({percentage:.2f}%)")

if __name__ == "__main__":
    main()
