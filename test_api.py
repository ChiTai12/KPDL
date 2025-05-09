import requests
import json

def test_recipe_by_ingredients(ingredients):
    """Kiểm tra API tìm kiếm công thức dựa trên nguyên liệu"""
    url = "http://127.0.0.1:5000/recommend_by_ingredients"
    data = {"ingredients": ",".join(ingredients)}
    
    print(f"\nTìm kiếm công thức với nguyên liệu: {', '.join(ingredients)}")
    
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        
        recipes = response.json()
        
        if isinstance(recipes, dict) and "error" in recipes:
            print(f"Lỗi: {recipes['error']}")
            return
        
        print(f"Tìm thấy {len(recipes)} công thức:")
        
        for i, recipe in enumerate(recipes, 1):
            print(f"\n{i}. {recipe['recipe_name']}")
            print(f"   Số nguyên liệu khớp: {recipe.get('matching_count', 'N/A')}")
            print(f"   Điểm khớp: {recipe.get('match_score', 0):.2f} ({int(recipe.get('match_score', 0) * 100)}%)")
            print(f"   Tỷ lệ nguyên liệu khớp: {recipe.get('ingredient_coverage', 0):.2f}")
            print(f"   Điểm tập phổ biến: {recipe.get('fp_score', 0):.2f}")
            print(f"   Điểm tổng hợp: {recipe.get('final_score', 0):.2f}")
            print(f"   Đánh giá: {recipe.get('rating', 0)}")
            
    except requests.exceptions.RequestException as e:
        print(f"Lỗi kết nối: {e}")
    except json.JSONDecodeError:
        print("Lỗi: Không thể phân tích phản hồi JSON")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")

if __name__ == "__main__":
    # Kiểm tra với các bộ nguyên liệu khác nhau
    test_cases = [
        ["apple", "cinnamon", "sugar"],
        ["chicken", "rice", "carrot"],
        ["flour", "butter", "sugar", "egg"],
        ["beef", "onion", "garlic"]
    ]
    
    for ingredients in test_cases:
        test_recipe_by_ingredients(ingredients)
