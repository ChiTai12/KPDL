import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from recipe_recommender import load_and_preprocess_data, find_ingredient_relationships

# Load và xử lý dữ liệu
print("Đang đọc và xử lý dữ liệu...")
df = load_and_preprocess_data()
print(f"Đã đọc {len(df)} công thức (sau khi loại bỏ trùng lặp)")

# Tìm mối quan hệ giữa các nguyên liệu với min_support = 0.005
print("Đang tìm mối quan hệ giữa các nguyên liệu...")
frequent_itemsets, _ = find_ingredient_relationships(df)
print(f"Đã tìm thấy {len(frequent_itemsets)} tập nguyên liệu phổ biến")

# Lọc ra các tập nguyên liệu có 1 phần tử (nguyên liệu đơn lẻ)
single_items = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 1)]
print(f"Số lượng nguyên liệu phổ biến (1-itemsets): {len(single_items)}")

# Sắp xếp theo độ hỗ trợ giảm dần
single_items = single_items.sort_values('support', ascending=False)

# In ra tất cả các nguyên liệu phổ biến
print("\nDanh sách tất cả các nguyên liệu phổ biến:")
for i, (_, row) in enumerate(single_items.iterrows(), 1):
    ingredient = list(row['itemsets'])[0]
    count = int(row['support'] * len(df))
    percentage = row['support'] * 100
    print(f"{i}. {ingredient}: {count} lần xuất hiện ({percentage:.2f}%)")

# In ra 20 nguyên liệu phổ biến nhất
print("\n20 nguyên liệu phổ biến nhất:")
for i, (_, row) in enumerate(single_items.head(20).iterrows(), 1):
    ingredient = list(row['itemsets'])[0]
    count = int(row['support'] * len(df))
    percentage = row['support'] * 100
    print(f"{i}. {ingredient}: {count} lần xuất hiện ({percentage:.2f}%)")

# Đếm số lần xuất hiện của mỗi nguyên liệu theo cách thông thường
all_ingredients = []
for ingredients in df['cleaned_ingredients']:
    all_ingredients.extend(ingredients)
ingredient_counts = pd.Series(all_ingredients).value_counts()

print("\n20 nguyên liệu phổ biến nhất (theo cách đếm thông thường):")
for i, (ingredient, count) in enumerate(ingredient_counts.head(20).items(), 1):
    percentage = count / len(df) * 100
    print(f"{i}. {ingredient}: {count} lần xuất hiện ({percentage:.2f}%)")
