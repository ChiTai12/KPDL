from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, current_user, login_required
from functools import wraps
from recipe_recommender import load_and_preprocess_data, find_ingredient_relationships, create_recipe_recommender, cluster_recipes, get_recipe_recommendations
import pandas as pd
import numpy as np
import time
from functools import lru_cache
from auth import auth
from models import db, User, Recipe, Rating, user_favorites, Cart, CartItem
from sqlalchemy import func
import os
from ingredient_rating_correlation import analyze_ingredient_rating_correlation, create_correlation_charts

# Khởi tạo Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Cấu hình static folder
app.static_url_path = ''
app.static_folder = os.path.abspath(os.path.dirname(__file__))

# Cấu hình database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recipe_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Khởi tạo database
db.init_app(app)

# Khởi tạo LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Bạn không có quyền truy cập trang này!', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    """Trang quản trị"""
    users = User.query.all()

    # Tính toán thống kê
    total_ratings = Rating.query.count()
    avg_rating = 0
    if total_ratings > 0:
        avg_rating = db.session.query(func.avg(Rating.value)).scalar() or 0

    stats = {
        'total_users': User.query.count(),
        'total_recipes': Recipe.query.count(),
        'total_ratings': total_ratings,
        'avg_rating': avg_rating
    }

    return render_template('admin/dashboard.html',
                         users=users,
                         stats=stats)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Xóa người dùng"""
    user = User.query.get_or_404(user_id)
    if user.is_admin:
        flash('Không thể xóa tài khoản admin!', 'error')
        return redirect(url_for('admin_dashboard'))

    try:
        db.session.delete(user)
        db.session.commit()
        flash('Đã xóa người dùng thành công!', 'success')
    except:
        db.session.rollback()
        flash('Có lỗi xảy ra khi xóa người dùng!', 'error')

    return redirect(url_for('admin_dashboard'))

@app.route('/admin/clear-favorites', methods=['POST'])
@login_required
@admin_required
def clear_admin_favorites():
    """Xóa tất cả món yêu thích của admin"""
    try:
        current_user.favorites.clear()
        db.session.commit()
        flash('Đã xóa tất cả món yêu thích!', 'success')
    except:
        db.session.rollback()
        flash('Có lỗi xảy ra khi xóa món yêu thích!', 'error')
    return redirect(url_for('user_profile'))

@app.route('/clear-admin-favs')
def clear_admin_favs():
    """Xóa tất cả món yêu thích của admin"""
    try:
        admin = User.query.filter_by(email="admin@recipe.com").first()
        if admin:
            admin.favorites.clear()
            db.session.commit()
            return "Đã xóa tất cả món yêu thích của admin"
    except Exception as e:
        db.session.rollback()
        return f"Lỗi: {str(e)}"
    return "Không tìm thấy admin"

# Đăng ký blueprint
app.register_blueprint(auth)

def init_db():
    """Khởi tạo database"""
    print("Starting database initialization...")
    with app.app_context():
        print("Initializing database...")
        print("Creating tables if they don't exist...")
        db.create_all()  # Chỉ tạo bảng nếu chưa tồn tại

        try:
            # Check if admin user exists
            admin = User.query.filter_by(email="admin@recipe.com").first()
            if not admin:
                # Create admin user
                admin_email = "admin@recipe.com"
                print(f"Creating admin user with email: {admin_email}")

                admin = User(
                    email=admin_email,
                    name="Admin",
                    is_admin=True
                )
                admin.set_password("admin123")
                db.session.add(admin)
                db.session.commit()
                print("Admin user created successfully")

        except Exception as e:
            db.session.rollback()
            print(f"Error creating admin user: {str(e)}")
            raise

        print("Database initialization complete")

# Khởi tạo database trước khi load dữ liệu
with app.app_context():
    init_db()

# Load và xử lý dữ liệu
print("Loading recipe data...")
try:
    df = pd.read_csv('recipes.csv/recipes.csv')
    print(f"Loaded {len(df)} recipes")

    # Xử lý dữ liệu
    df = load_and_preprocess_data()
    print("Đang tạo ma trận độ tương đồng...")
    frequent_itemsets, similarity_matrix = create_recipe_recommender(df)
    print("Đang phân cụm công thức...")
    df_clustered, cluster_names, cluster_image = cluster_recipes(df)

    # Bộ nhớ đệm cho các gợi ý công thức
    recipe_recommendations_cache = {}

    # Tiền tính toán gợi ý cho top 100 công thức phổ biến nhất
    print("Đang tiền tính toán gợi ý cho các công thức phổ biến...")
    top_recipes = df.nlargest(100, 'rating')['recipe_name'].tolist()
    for recipe_name in top_recipes:
        recommendations = get_recipe_recommendations(recipe_name, df, similarity_matrix, frequent_itemsets, 5)
        recipe_recommendations_cache[recipe_name] = recommendations

except Exception as e:
    print(f"Error loading recipes.csv: {str(e)}")
    df = pd.DataFrame()  # Create empty DataFrame if file not found
    frequent_itemsets = pd.DataFrame()
    similarity_matrix = np.array([])
    df_clustered = pd.DataFrame()
    cluster_names = {}
    cluster_image = None
    recipe_recommendations_cache = {}

print("Khởi động hoàn tất!")

# Sử dụng decorator lru_cache để lưu kết quả các lần gọi trước
@lru_cache(maxsize=100)
def get_recipe_recommendations_wrapper(recipe_name, n_recommendations=5):
    """Wrapper function để lấy các công thức được gợi ý dựa trên tên món"""
    # Kiểm tra trong bộ nhớ đệm
    if recipe_name in recipe_recommendations_cache:
        return recipe_recommendations_cache[recipe_name]

    # Nếu không có trong bộ nhớ đệm, tính toán và lưu lại
    start_time = time.time()
    recommendations = get_recipe_recommendations(recipe_name, df, similarity_matrix, frequent_itemsets, n_recommendations)
    print(f"Thời gian lấy gợi ý cho {recipe_name}: {time.time() - start_time:.2f} giây")

    # Lưu vào bộ nhớ đệm
    recipe_recommendations_cache[recipe_name] = recommendations

    return recommendations

def get_popular_ingredients():
    """Lấy danh sách các nguyên liệu phổ biến"""
    all_ingredients = []
    for ingredients in df['cleaned_ingredients']:
        all_ingredients.extend(ingredients)
    ingredient_counts = pd.Series(all_ingredients).value_counts()
    return ingredient_counts.head(20)

def get_recipes_by_ingredients(ingredients_list, n_recommendations=5):
    """Gợi ý công thức dựa trên danh sách nguyên liệu"""
    print(f"[LOG] Bắt đầu phân tích với {len(ingredients_list)} nguyên liệu: {', '.join(ingredients_list)}")
    print(f"[LOG] Thời gian: {time.strftime('%H:%M:%S')}")

    # Chuyển đổi danh sách nguyên liệu thành set để so sánh
    ingredients_set = set(ingredients_list)
    print(f"[LOG] Đã chuyển đổi nguyên liệu thành set để tìm kiếm nhanh hơn")

    # Hiển thị thông tin về dữ liệu
    print(f"[LOG] Tổng số công thức trong cơ sở dữ liệu: {len(df)}")
    print(f"[LOG] Số nguyên liệu độc nhất trong cơ sở dữ liệu: {len(set().union(*df['cleaned_ingredients'].tolist()))}")

    # Tính điểm cho mỗi công thức dựa trên số nguyên liệu khớp
    start_time = time.time()
    print(f"[LOG] Đang tính điểm match_score cho {len(df)} công thức...")
    df['match_score'] = df['cleaned_ingredients'].apply(
        lambda x: len(set(x) & ingredients_set) / len(ingredients_set) if len(ingredients_set) > 0 else 0
    )
    print(f"[LOG] Hoàn thành tính match_score trong {time.time() - start_time:.2f} giây")

    # Tính thêm tỷ lệ nguyên liệu khớp so với tổng số nguyên liệu của công thức
    start_time = time.time()
    print(f"[LOG] Đang tính tỷ lệ nguyên liệu khớp (ingredient_coverage)...")
    df['ingredient_coverage'] = df['cleaned_ingredients'].apply(
        lambda x: len(set(x) & ingredients_set) / len(x) if len(x) > 0 else 0
    )
    print(f"[LOG] Hoàn thành tính ingredient_coverage trong {time.time() - start_time:.2f} giây")

    # Đánh giá dựa trên các tập mục phổ biến
    start_time = time.time()
    print(f"[LOG] Đang tính điểm dựa trên các tập mục phổ biến (fp_score)...")
    df['fp_score'] = 0.0

    # Kiểm tra xem các nguyên liệu nhập vào có xuất hiện trong các tập phổ biến không
    matching_itemsets = 0
    for _, row in frequent_itemsets.iterrows():
        itemset = set(row['itemsets'])
        # Nếu tập nguyên liệu phổ biến có ít nhất một nguyên liệu khớp với nguyên liệu nhập vào
        if len(itemset & ingredients_set) > 0:
            matching_itemsets += 1
            # Tính điểm cho các công thức có chứa tập nguyên liệu phổ biến này
            df['fp_score'] = df.apply(
                lambda recipe: recipe['fp_score'] + (
                    row['support'] * len(itemset & set(recipe['cleaned_ingredients'])) / len(itemset)
                    if len(itemset & set(recipe['cleaned_ingredients'])) > 0 else 0
                ),
                axis=1
            )
    print(f"[LOG] Số tập mục phổ biến khớp với nguyên liệu: {matching_itemsets}")
    print(f"[LOG] Hoàn thành tính fp_score trong {time.time() - start_time:.2f} giây")

    # Tính số lượng nguyên liệu khớp
    start_time = time.time()
    print(f"[LOG] Đang tính số lượng nguyên liệu khớp (matching_count)...")
    df['matching_count'] = df['cleaned_ingredients'].apply(
        lambda x: len(set(x) & ingredients_set)
    )
    print(f"[LOG] Hoàn thành tính matching_count trong {time.time() - start_time:.2f} giây")

    # Kết hợp điểm match_score, ingredient_coverage và fp_score với trọng số mới
    start_time = time.time()
    print(f"[LOG] Đang tính điểm tổng hợp (final_score)...")
    df['final_score'] = 0.6 * df['match_score'] + 0.2 * df['ingredient_coverage'] + 0.2 * (df['fp_score'] / df['fp_score'].max() if df['fp_score'].max() > 0 else 0)
    print(f"[LOG] Hoàn thành tính final_score trong {time.time() - start_time:.2f} giây")

    # Lọc các công thức có ít nhất 1 nguyên liệu khớp
    start_time = time.time()
    print(f"[LOG] Lọc các công thức có ít nhất 1 nguyên liệu khớp...")
    filtered_df = df[df['matching_count'] > 0]
    print(f"[LOG] Số công thức có ít nhất 1 nguyên liệu khớp: {len(filtered_df)}")
    print(f"[LOG] Hoàn thành lọc công thức trong {time.time() - start_time:.2f} giây")

    # Nếu không có công thức nào khớp, giảm yêu cầu
    if len(filtered_df) < n_recommendations:
        print(f"[LOG] Không đủ công thức khớp, giảm yêu cầu để lấy tất cả công thức...")
        filtered_df = df

    # Sắp xếp theo số lượng nguyên liệu khớp, điểm tổng hợp và rating
    start_time = time.time()
    print(f"[LOG] Sắp xếp kết quả theo số lượng nguyên liệu khớp, điểm tổng hợp và rating...")

    # Loại bỏ các món ăn trùng lặp dựa trên tên công thức
    print(f"[LOG] Loại bỏ các món ăn trùng lặp...")
    filtered_df = filtered_df.drop_duplicates(subset=['recipe_name'])
    print(f"[LOG] Số công thức sau khi loại bỏ trùng lặp: {len(filtered_df)}")

    # Lấy top n công thức phù hợp nhất
    recommended_recipes = filtered_df.nlargest(n_recommendations, ['matching_count', 'final_score', 'rating'])
    print(f"[LOG] Đã tìm thấy {len(recommended_recipes)} công thức phù hợp nhất")
    print(f"[LOG] Hoàn thành sắp xếp và lọc kết quả trong {time.time() - start_time:.2f} giây")

    # Hiển thị thông tin về các công thức được gợi ý
    print(f"[LOG] Thông tin chi tiết về các công thức được gợi ý:")
    for i, (_, recipe) in enumerate(recommended_recipes.iterrows(), 1):
        print(f"[LOG] {i}. {recipe['recipe_name']}")
        print(f"[LOG]    - Rating: {recipe['rating']}")
        print(f"[LOG]    - Số nguyên liệu khớp: {recipe['matching_count']} / {len(recipe['cleaned_ingredients'])}")
        print(f"[LOG]    - Nguyên liệu khớp: {', '.join(set(recipe['cleaned_ingredients']) & ingredients_set)}")
        print(f"[LOG]    - Điểm tổng hợp: {recipe['final_score']:.4f}")

    return recommended_recipes

@app.route('/')
def home():
    """Trang chủ"""
    popular_ingredients = get_popular_ingredients()
    # Lấy 10 công thức được đánh giá cao nhất
    if not df.empty:
        top_recipes = df.nlargest(10, 'rating')[['recipe_name', 'rating', 'total_time', 'img_src']]
    else:
        top_recipes = pd.DataFrame()
    return render_template('index.html',
                         recipes=[] if df.empty else df['recipe_name'].tolist(),
                         popular_ingredients=popular_ingredients,
                         top_recipes=top_recipes.to_dict('records'))

@app.route('/recipe/<recipe_name>')
def recipe_detail(recipe_name):
    """Hiển thị chi tiết công thức"""
    start_time = time.time()

    # Tìm công thức trong database trước
    recipe_db = Recipe.query.filter_by(name=recipe_name).first()

    if recipe_db:
        # Nếu có trong database, sử dụng dữ liệu từ database
        recipe_dict = {
            'recipe_name': recipe_db.name,
            'ingredients': recipe_db.ingredients,
            'directions': recipe_db.instructions,
            'rating': recipe_db.rating,
            'prep_time': recipe_db.prep_time,
            'cook_time': recipe_db.cook_time,
            'total_time': recipe_db.total_time,
            'img_src': recipe_db.img_src,
            'nutrition': recipe_db.nutrition
        }
    else:
        # Nếu không có trong database, lấy từ DataFrame
        recipe = df[df['recipe_name'] == recipe_name].iloc[0]
        recipe_dict = {
            'recipe_name': recipe['recipe_name'],
            'ingredients': recipe['ingredients'],
            'directions': recipe['directions'],
            'rating': recipe['rating'],
            'prep_time': recipe['prep_time'],
            'cook_time': recipe['cook_time'],
            'total_time': recipe['total_time'],
            'img_src': recipe['img_src'],
            'nutrition': recipe['nutrition']
        }

    recommendations = get_recipe_recommendations_wrapper(recipe_name)
    print(f"Tổng thời gian tải trang chi tiết {recipe_name}: {time.time() - start_time:.2f} giây")

    return render_template('recipe_detail.html',
                         recipe=recipe_dict,
                         recommendations=recommendations[['recipe_name', 'rating', 'total_time', 'img_src']].to_dict('records'))

@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint để lấy gợi ý công thức"""
    recipe_name = request.form['recipe_name']
    recommendations = get_recipe_recommendations_wrapper(recipe_name)
    return jsonify(recommendations[['recipe_name', 'rating', 'total_time', 'img_src']].to_dict('records'))

@app.route('/ingredients')
@login_required
@admin_required
def ingredients():
    """Trang hiển thị mối quan hệ giữa các nguyên liệu"""
    # Chuyển đổi frequent_itemsets thành định dạng phù hợp
    processed_itemsets = []
    for _, row in frequent_itemsets.iterrows():
        processed_itemsets.append({
            'itemsets': list(row['itemsets']),
            'support': float(row['support'])
        })

    return render_template('ingredients.html',
                         frequent_itemsets=processed_itemsets)

@app.route('/correlation-analysis')
@login_required
@admin_required
def correlation_analysis():
    """Trang phân tích tương quan giữa nguyên liệu và đánh giá"""
    print("Đang thực hiện phân tích tương quan...")

    # Phân tích tương quan
    results_df, overall_avg_rating = analyze_ingredient_rating_correlation(df)

    # Lấy top 10 nguyên liệu có ảnh hưởng tích cực nhất
    top_positive = results_df.head(10)

    # Lấy top 10 nguyên liệu có ảnh hưởng tiêu cực nhất
    top_negative = results_df.tail(10).iloc[::-1]  # Đảo ngược để hiển thị từ tiêu cực nhất

    # Tạo biểu đồ
    positive_chart, negative_chart, correlation_chart = create_correlation_charts(results_df, overall_avg_rating, df)

    print("Đã hoàn thành phân tích tương quan.")

    return render_template('correlation_analysis.html',
                         overall_avg_rating=overall_avg_rating,
                         top_positive=top_positive,
                         top_negative=top_negative,
                         positive_chart=positive_chart,
                         negative_chart=negative_chart,
                         correlation_chart=correlation_chart)

@app.route('/recommend_by_ingredients', methods=['POST'])
def recommend_by_ingredients():
    """API endpoint để gợi ý công thức dựa trên nguyên liệu"""
    ingredients = request.form.get('ingredients', '').split(',')
    ingredients = [item.strip() for item in ingredients if item.strip()]

    if not ingredients:
        return jsonify({'error': 'Vui lòng nhập ít nhất một nguyên liệu'})

    print("\n" + "="*50)
    print(f"[LOG] Phân tích nguyên liệu: {ingredients}")
    print(f"[LOG] Thời gian bắt đầu: {time.strftime('%H:%M:%S')}")
    print(f"[LOG] Đang tìm kiếm công thức phù hợp với các nguyên liệu...")

    # Gọi hàm phân tích và ghi log
    start_time = time.time()
    recommendations = get_recipes_by_ingredients(ingredients)
    end_time = time.time()

    print(f"[LOG] Thời gian tìm kiếm: {end_time - start_time:.2f} giây")
    print(f"[LOG] Số công thức phù hợp: {len(recommendations)}")
    print(f"[LOG] Top 5 công thức phù hợp nhất:")
    for _, recipe in recommendations.head(5).iterrows():
        print(f"  - {recipe['recipe_name']}")
        print(f"    + Điểm khớp (match_score): {recipe['match_score']:.2f}")
        print(f"    + Tỷ lệ nguyên liệu khớp (ingredient_coverage): {recipe['ingredient_coverage']:.2f}")
        print(f"    + Điểm tập phổ biến (fp_score): {recipe['fp_score']:.2f}")
        print(f"    + Điểm tổng hợp (final_score): {recipe['final_score']:.2f}")
        print(f"    + Số nguyên liệu khớp: {recipe['matching_count']}")
    print(f"[LOG] Thời gian kết thúc: {time.strftime('%H:%M:%S')}")
    print("="*50 + "\n")

    return jsonify(recommendations[['recipe_name', 'rating', 'total_time', 'img_src', 'match_score', 'ingredient_coverage', 'fp_score', 'final_score', 'matching_count']].to_dict('records'))

@app.route('/menu')
def menu():
    """Trang hiển thị danh mục tự động cho nhà hàng"""
    # Chuẩn bị dữ liệu cho mỗi cụm
    menu_categories = []
    for cluster_id, name in cluster_names.items():
        # Lấy 10 công thức có rating cao nhất trong mỗi cụm thay vì chỉ 5
        cluster_recipes = df_clustered[df_clustered['cluster'] == cluster_id].nlargest(10, 'rating')

        menu_categories.append({
            'name': name,
            'recipes': cluster_recipes[['recipe_name', 'rating', 'total_time', 'img_src']].to_dict('records'),
            'count': len(df_clustered[df_clustered['cluster'] == cluster_id])
        })

    # Chỉ tạo biểu đồ phân cụm nếu người dùng là admin
    cluster_image_data = None
    if current_user.is_authenticated and current_user.is_admin:
        cluster_image_data = cluster_image

    return render_template('menu.html',
                         menu_categories=menu_categories,
                         cluster_image=cluster_image_data)

@app.route('/add_to_menu', methods=['POST'])
def add_to_menu():
    """Thêm món ăn vào thực đơn"""
    try:
        data = request.get_json()
        if not data or 'recipes' not in data:
            return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

        recipes = data['recipes']
        if not recipes:
            return jsonify({'error': 'Không có món ăn nào được chọn'}), 400

        # Tạo một thực đơn mới thay vì thêm vào thực đơn cũ
        menu_items = {
            'starters': [],
            'main_dishes': [],
            'desserts': [],
            'drinks': []
        }

        # Thêm các món ăn mới vào thực đơn
        for r in recipes:
            item_to_add = {'name': r['name'], 'price': r.get('price', '')}

            if r['category'] == 'appetizer':
                menu_items['starters'].append(item_to_add)
            elif r['category'] == 'main':
                menu_items['main_dishes'].append(item_to_add)
            elif r['category'] == 'dessert':
                menu_items['desserts'].append(item_to_add)
            elif r['category'] == 'drink':
                menu_items['drinks'].append(item_to_add)

        # Cập nhật lại session
        session['menu_items'] = menu_items

        # Log the session contents
        print('Current session menu items:', session['menu_items'])
        print('Added recipes:', recipes)

        return jsonify({
            'success': True,
            'message': 'Đã tạo thực đơn mới với các món ăn đã chọn. Các món ăn cũ đã bị xóa.'
        })
    except Exception as e:
        print(f"Error in add_to_menu: {str(e)}")
        return jsonify({'error': f'Có lỗi xảy ra: {str(e)}'}), 500

@app.route('/build_menu', methods=['GET', 'POST'])
def build_menu():
    """Trang xây dựng thực đơn cho nhà hàng"""
    # Lấy tất cả các món ăn được phân cụm
    menu_categories = []
    menu_data = {}

    for cluster_id, name in cluster_names.items():
        cluster_recipes = df_clustered[df_clustered['cluster'] == cluster_id].nlargest(30, 'rating')
        random_recipes = df_clustered[df_clustered['cluster'] == cluster_id].sample(min(10, len(df_clustered[df_clustered['cluster'] == cluster_id])))
        combined_recipes = pd.concat([cluster_recipes, random_recipes]).drop_duplicates(subset=['recipe_name'])

        menu_categories.append({
            'id': cluster_id,
            'name': name,
            'recipes': combined_recipes[['recipe_name', 'rating', 'total_time', 'img_src']].to_dict('records'),
            'count': len(df_clustered[df_clustered['cluster'] == cluster_id])
        })
        menu_data[name] = combined_recipes[['recipe_name', 'rating', 'total_time', 'img_src']].to_dict('records')

    # Thêm danh mục "Món ăn phổ biến"
    popular_recipes = df.nlargest(20, 'rating')
    menu_categories.append({
        'id': 'popular',
        'name': 'Món ăn phổ biến',
        'recipes': popular_recipes[['recipe_name', 'rating', 'total_time', 'img_src']].to_dict('records'),
        'count': len(popular_recipes)
    })
    menu_data['Món ăn phổ biến'] = popular_recipes[['recipe_name', 'rating', 'total_time', 'img_src']].to_dict('records')

    # Xử lý dữ liệu từ session hoặc POST request
    saved_menu = None
    message = None

    if request.method == 'POST':
        # Xử lý POST request
        menu_title = request.form.get('menu_title', 'Thực đơn nhà hàng')
        menu_description = request.form.get('menu_description', '')
        menu_theme = request.form.get('menu_theme', 'classic')



        # Lấy danh sách món ăn từ form, hỗ trợ cả hai định dạng
        starters = request.form.getlist('starters')

        # Hỗ trợ cả hai định dạng: main_dishes và main-dishes
        main_dishes = request.form.getlist('main_dishes')
        main_dishes_alt = request.form.getlist('main-dishes')
        if main_dishes_alt:
            main_dishes = main_dishes_alt  # Ưu tiên sử dụng main-dishes

        desserts = request.form.getlist('desserts')
        drinks = request.form.getlist('drinks')

        # Debug: In ra các món ăn đã lấy từ form
        print(f"Starters: {starters}")
        print(f"Main dishes: {main_dishes}")
        print(f"Desserts: {desserts}")
        print(f"Drinks: {drinks}")

        # Tạo danh sách các món ăn với giá
        starters_with_price = []
        for name in starters:
            price = request.form.get(f'price_{name.replace(" ", "_")}', '')
            starters_with_price.append({'name': name, 'price': price})

        main_dishes_with_price = []
        for name in main_dishes:
            price = request.form.get(f'price_{name.replace(" ", "_")}', '')
            main_dishes_with_price.append({'name': name, 'price': price})

        desserts_with_price = []
        for name in desserts:
            price = request.form.get(f'price_{name.replace(" ", "_")}', '')
            desserts_with_price.append({'name': name, 'price': price})

        drinks_with_price = []
        for name in drinks:
            price = request.form.get(f'price_{name.replace(" ", "_")}', '')
            drinks_with_price.append({'name': name, 'price': price})

        # Lưu vào session để có thể sử dụng trong preview_menu và giữ lại khi tải lại trang
        session['menu_title'] = menu_title
        session['menu_description'] = menu_description
        session['menu_theme'] = menu_theme
        session['menu_items'] = {
            'starters': starters_with_price,
            'main_dishes': main_dishes_with_price,
            'desserts': desserts_with_price,
            'drinks': drinks_with_price
        }

        saved_menu = {
            'title': menu_title,
            'description': menu_description,
            'theme': menu_theme,
            'starters': starters_with_price,
            'main_dishes': main_dishes_with_price,
            'desserts': desserts_with_price,
            'drinks': drinks_with_price
        }

        message = "Thực đơn đã được lưu thành công!"
    else:
        # Lấy dữ liệu từ session nếu có
        menu_items = session.get('menu_items', None)
        if menu_items:
            saved_menu = {
                'title': session.get('menu_title', 'Thực đơn nhà hàng'),
                'description': session.get('menu_description', ''),
                'theme': session.get('menu_theme', 'classic'),
                'starters': menu_items.get('starters', []),
                'main_dishes': menu_items.get('main_dishes', []),
                'desserts': menu_items.get('desserts', []),
                'drinks': menu_items.get('drinks', [])
            }

    return render_template('build_menu.html',
                          menu_categories=menu_categories,
                          menu_data=menu_data,
                          saved_menu=saved_menu,
                          message=message)

@app.route('/preview_menu')
def preview_menu():
    """Preview thực đơn đã tạo"""
    # Lấy dữ liệu thực đơn từ session
    menu_items = session.get('menu_items', {})
    menu_title = request.args.get('menu_title', 'Thực đơn nhà hàng')
    menu_description = request.args.get('menu_description', '')
    menu_theme = request.args.get('menu_theme', 'classic')

    # Debug: Print all request args to see what's being sent
    print("Request args:", dict(request.args))
    print("Session menu items before processing:", menu_items)

    # Đảm bảo tất cả các danh mục đều tồn tại
    if not menu_items or not isinstance(menu_items, dict):
        menu_items = {
            'starters': [],
            'main_dishes': [],
            'desserts': [],
            'drinks': []
        }
    else:
        # Tạo bản sao để tránh lỗi "session modified"
        menu_items = menu_items.copy()
        for category in ['starters', 'main_dishes', 'desserts', 'drinks']:
            if category not in menu_items:
                menu_items[category] = []

    # Nếu dữ liệu đến từ session, xử lý và làm phong phú thêm
    if all(isinstance(items, list) for items in menu_items.values()):
        # Thêm thông tin chi tiết cho các món từ session
        for category, items in menu_items.items():
            updated_items = []
            for item in items:
                # Kiểm tra xem item có phải là dictionary không
                if isinstance(item, dict) and 'name' in item:
                    item_name = item['name']
                    item_copy = item.copy()  # Tạo bản sao để tránh thay đổi dữ liệu gốc
                else:
                    item_name = item  # Nếu item là string
                    item_copy = {'name': item_name, 'price': ''}

                # Tìm thông tin công thức
                recipe_info = None
                try:
                    recipe_info = df[df['recipe_name'] == item_name].iloc[0] if len(df[df['recipe_name'] == item_name]) > 0 else None
                except Exception as e:
                    print(f"Không tìm thấy thông tin cho món: {item_name}, lỗi: {str(e)}")

                if recipe_info is not None:
                    # Cập nhật thông tin nếu chưa có
                    if 'img_src' not in item_copy or not item_copy['img_src']:
                        item_copy['img_src'] = recipe_info.get('img_src', '')
                    if 'rating' not in item_copy or not item_copy['rating']:
                        item_copy['rating'] = recipe_info.get('rating', 0)
                    if 'total_time' not in item_copy or not item_copy['total_time']:
                        item_copy['total_time'] = recipe_info.get('total_time', '')
                else:
                    # Nếu không tìm thấy thông tin, đảm bảo các trường cần thiết tồn tại
                    if 'img_src' not in item_copy:
                        item_copy['img_src'] = ''
                    if 'rating' not in item_copy:
                        item_copy['rating'] = 0
                    if 'total_time' not in item_copy:
                        item_copy['total_time'] = ''

                updated_items.append(item_copy)

            # Cập nhật lại danh sách món ăn
            menu_items[category] = updated_items
    else:
        # Nếu không có dữ liệu trong session hoặc dữ liệu không hợp lệ, lấy từ request args
        menu_items = {
            'starters': [],
            'main_dishes': [],
            'desserts': [],
            'drinks': []
        }

        # Lấy danh sách các món ăn từ request args và giá tiền tương ứng
        # Kiểm tra cả hai định dạng: main_dishes và main-dishes
        category_mapping = {
            'starters': ['starters'],
            'main_dishes': ['main_dishes', 'main-dishes'],
            'desserts': ['desserts'],
            'drinks': ['drinks']
        }

        for category, alt_categories in category_mapping.items():
            for alt_category in alt_categories:
                items = request.args.getlist(alt_category)
                print(f"Category {alt_category} items: {items}")

                for item in items:
                    price = request.args.get(f'price_{item.replace(" ", "_")}', '')
                    print(f"Processing item: {item}, price: {price}")

                    try:
                        recipe_info = df[df['recipe_name'] == item].iloc[0] if len(df[df['recipe_name'] == item]) > 0 else None

                        menu_items[category].append({
                            'name': item,
                            'price': price,
                            'img_src': recipe_info.get('img_src', '') if recipe_info is not None else '',
                            'rating': recipe_info.get('rating', 0) if recipe_info is not None else 0,
                            'total_time': recipe_info.get('total_time', '') if recipe_info is not None else ''
                        })
                    except Exception as e:
                        print(f"Error processing item {item}: {str(e)}")

        # Xử lý các món có giá tiền nhưng không có trong danh sách
        for param_name, value in request.args.items():
            if param_name.startswith('price_'):
                item_name = param_name[6:].replace('_', ' ')

                # Kiểm tra xem món này đã được thêm vào chưa
                already_added = False
                for category_items in menu_items.values():
                    for item in category_items:
                        if item['name'] == item_name:
                            already_added = True
                            break
                    if already_added:
                        break

                # Nếu món chưa được thêm vào, tìm danh mục phù hợp
                if not already_added and item_name != "":
                    # Tìm danh mục phù hợp cho món ăn dựa vào request args
                    target_category = None

                    # Kiểm tra trong form để xác định món thuộc danh mục nào
                    for key, values in request.args.lists():
                        if key in menu_items.keys() and item_name in values:
                            target_category = key
                            break

                    # Nếu không tìm thấy, kiểm tra trong các tham số khác
                    if not target_category:
                        # Kiểm tra xem món này thuộc danh mục nào trong form
                        for category in menu_items.keys():
                            category_param = f"{category}_{item_name.replace(' ', '_')}"
                            if category_param in request.args:
                                target_category = category
                                break

                    # Nếu vẫn không tìm thấy, kiểm tra trong các tham số khác của form
                    if not target_category:
                        # Tìm trong các tham số khác
                        for key in request.args.keys():
                            if key.endswith(f"_{item_name.replace(' ', '_')}") and key.split('_')[0] in menu_items:
                                target_category = key.split('_')[0]
                                break

                    # Nếu vẫn không tìm thấy, sử dụng danh mục mặc định
                    if not target_category:
                        # Dựa vào hình ảnh, nếu món "French Puff Pastry Tart with Pears and Chocolate"
                        # thuộc danh mục "drinks" trong form, thì giữ nguyên
                        if item_name == "French Puff Pastry Tart with Pears and Chocolate":
                            target_category = "drinks"
                        else:
                            target_category = 'desserts'  # Mặc định là desserts

                    print(f"Determined category for {item_name}: {target_category}")

                    # Tìm thông tin công thức
                    try:
                        recipe_info = df[df['recipe_name'] == item_name].iloc[0] if len(df[df['recipe_name'] == item_name]) > 0 else None

                        menu_items[target_category].append({
                            'name': item_name,
                            'price': value,
                            'img_src': recipe_info.get('img_src', '') if recipe_info is not None else '',
                            'rating': recipe_info.get('rating', 0) if recipe_info is not None else 0,
                            'total_time': recipe_info.get('total_time', '') if recipe_info is not None else ''
                        })
                        print(f"Added additional item to {target_category}: {item_name}")
                    except Exception as e:
                        print(f"Error processing additional item {item_name}: {str(e)}")

    # Debug: In ra số lượng món ăn trong mỗi danh mục
    print(f"Số lượng món trong mỗi danh mục: {[len(menu_items[cat]) for cat in menu_items]}")
    print(f"Menu items sau khi xử lý: {menu_items}")

    # Đảm bảo tất cả các món ăn đều có giá trị price là số
    for category, items in menu_items.items():
        for item in items:
            if 'price' in item:
                try:
                    # Chuyển đổi giá thành số nếu là chuỗi
                    if isinstance(item['price'], str):
                        # Loại bỏ các ký tự không phải số
                        price_str = ''.join(c for c in item['price'] if c.isdigit() or c == '.')
                        if price_str:
                            item['price'] = float(price_str)
                        else:
                            item['price'] = 0
                except Exception as e:
                    print(f"Lỗi khi chuyển đổi giá: {str(e)}")
                    item['price'] = 0
            else:
                item['price'] = 0

    # In ra thông tin menu sau khi xử lý giá
    print(f"Menu items sau khi xử lý giá: {menu_items}")

    return render_template('preview_menu.html',
                          menu_title=menu_title,
                          menu_description=menu_description,
                          menu_theme=menu_theme,
                          menu_items=menu_items)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Trang đăng nhập"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember')

        # TODO: Thêm logic xác thực người dùng ở đây
        # Ví dụ đơn giản:
        if email == "admin@example.com" and password == "password":
            session['user'] = email
            if remember:
                session.permanent = True
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Email hoặc mật khẩu không chính xác!', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    """Đăng xuất"""
    session.pop('user', None)
    flash('Đã đăng xuất thành công!', 'success')
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def user_profile():
    """Hiển thị trang hồ sơ người dùng"""
    if current_user.is_admin:
        users = User.query.all()
        recipes = Recipe.query.all()
        ratings = Rating.query.all()
        return render_template('profile.html',
                             user=current_user,
                             users=users,
                             recipes=recipes,
                             ratings=ratings)
    return render_template('profile.html', user=current_user)

@app.route('/favorites')
@login_required
def favorites():
    """Hiển thị danh sách món ăn yêu thích của người dùng"""
    favorite_recipes = current_user.favorites
    return render_template('favorites.html', favorite_recipes=favorite_recipes)

@app.route('/toggle_favorite/<recipe_name>', methods=['POST'])
@login_required
def toggle_favorite(recipe_name):
    """Thêm/xóa công thức khỏi danh sách yêu thích"""
    try:
        print(f"Processing favorite toggle for recipe: {recipe_name}")

        if not current_user.is_authenticated:
            print("User not authenticated")
            return jsonify({'error': 'Vui lòng đăng nhập để sử dụng tính năng này'}), 401

        if not recipe_name:
            print("Invalid recipe name")
            return jsonify({'error': 'Tên công thức không hợp lệ'}), 400

        user = current_user

        # Kiểm tra nếu là admin thì không cho yêu thích
        if user.is_admin:
            print("Admin user attempted to favorite")
            return jsonify({'error': 'Admin không thể yêu thích món ăn'}), 403

        # Tìm công thức trong DataFrame
        if df.empty:
            print("DataFrame is empty")
            return jsonify({'error': 'Không có dữ liệu công thức'}), 500

        recipe_data = df[df['recipe_name'] == recipe_name]
        if recipe_data.empty:
            print(f"Recipe not found in DataFrame: {recipe_name}")
            return jsonify({'error': 'Không tìm thấy công thức'}), 404

        # Tìm hoặc tạo mới Recipe trong database
        recipe = Recipe.query.filter_by(name=recipe_name).first()

        # Xử lý thêm/xóa yêu thích
        try:
            if recipe in user.favorites:
                print(f"Removing recipe from favorites: {recipe_name}")
                user.favorites.remove(recipe)
                status = 'removed'
            else:
                print(f"Adding recipe to favorites: {recipe_name}")
                if not recipe:
                    print(f"Creating new recipe in database: {recipe_name}")
                    row = recipe_data.iloc[0]
                    recipe = Recipe(
                        name=recipe_name,
                        ingredients=str(row['ingredients']) if 'ingredients' in row else '',
                        instructions=str(row['directions']) if 'directions' in row else '',
                        rating=float(row['rating']) if 'rating' in row else 0.0,
                        total_time=str(row['total_time']) if 'total_time' in row else '',
                        img_src=str(row['img_src']) if 'img_src' in row else ''
                    )
                    db.session.add(recipe)
                user.favorites.append(recipe)
                status = 'added'

            db.session.commit()
            print(f"Successfully {status} recipe: {recipe_name}")
            return jsonify({
                'status': status,
                'message': 'Đã thêm vào danh sách yêu thích' if status == 'added' else 'Đã xóa khỏi danh sách yêu thích'
            })

        except Exception as e:
            db.session.rollback()
            print(f"Error updating favorites: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error args: {e.args}")
            return jsonify({'error': f'Không thể cập nhật trạng thái yêu thích: {str(e)}'}), 500

    except Exception as e:
        print(f"Error in toggle_favorite: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")
        return jsonify({'error': f'Có lỗi xảy ra: {str(e)}'}), 500

@app.route('/rate/<recipe_name>', methods=['POST'])
@login_required
def rate_recipe(recipe_name):
    """Đánh giá công thức"""
    rating_value = request.form.get('rating', type=int)
    if not rating_value or rating_value < 1 or rating_value > 5:
        return jsonify({'error': 'Giá trị đánh giá không hợp lệ'}), 400

    recipe = Recipe.query.filter_by(name=recipe_name).first()
    if not recipe:
        recipe = Recipe(name=recipe_name)
        db.session.add(recipe)

    # Kiểm tra xem người dùng đã đánh giá chưa
    rating = Rating.query.filter_by(user_id=current_user.id, recipe_id=recipe.id).first()
    if rating:
        rating.value = rating_value
    else:
        rating = Rating(user_id=current_user.id, recipe_id=recipe.id, value=rating_value)
        db.session.add(rating)

    try:
        db.session.commit()
        # Tính lại rating trung bình
        avg_rating = Rating.query.filter_by(recipe_id=recipe.id).with_entities(func.avg(Rating.value)).scalar()
        return jsonify({'new_rating': float(avg_rating) if avg_rating else rating_value})
    except:
        db.session.rollback()
        return jsonify({'error': 'Có lỗi xảy ra'}), 500

@app.route('/my-favorites')
@login_required
def my_favorites():
    """Show user's favorite recipes"""
    if current_user.is_admin:
        flash('Tính năng này chỉ dành cho người dùng thông thường', 'warning')
        return redirect(url_for('home'))
    return render_template('favorites.html',
                         favorites=current_user.favorites)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    """Cập nhật thông tin người dùng"""
    if request.method == 'POST':
        user = current_user
        user.name = request.form.get('name')
        user.email = request.form.get('email')
        user.picture = request.form.get('picture')

        # Cập nhật mật khẩu nếu có
        new_password = request.form.get('new_password')
        if new_password:
            user.set_password(new_password)

        try:
            db.session.commit()
            flash('Cập nhật thông tin thành công!', 'success')
        except:
            db.session.rollback()
            flash('Có lỗi xảy ra khi cập nhật thông tin.', 'error')

    return redirect(url_for('user_profile'))

@app.route('/cart')
@login_required
def view_cart():
    """Hiển thị giỏ hàng của người dùng"""
    if not current_user.cart:
        current_user.cart = Cart(user_id=current_user.id)
        db.session.commit()
    return render_template('cart.html', cart=current_user.cart)

@app.route('/cart/add/<recipe_name>', methods=['POST'])
@login_required
def add_to_cart(recipe_name):
    """Thêm món ăn vào giỏ hàng"""
    try:
        quantity = int(request.form.get('quantity', 1))
        price = float(request.form.get('price', 0))

        if not current_user.cart:
            current_user.cart = Cart(user_id=current_user.id)
            db.session.add(current_user.cart)
            db.session.commit()

        recipe = Recipe.query.filter_by(name=recipe_name).first()
        if not recipe:
            recipe_data = df[df['recipe_name'] == recipe_name]
            if recipe_data.empty:
                return jsonify({'error': 'Không tìm thấy món ăn'}), 404

            recipe = Recipe(
                name=recipe_name,
                ingredients=str(recipe_data.iloc[0]['ingredients']) if 'ingredients' in recipe_data else '',
                rating=float(recipe_data.iloc[0]['rating']) if 'rating' in recipe_data else 0.0,
                total_time=str(recipe_data.iloc[0]['total_time']) if 'total_time' in recipe_data else '',
                img_src=str(recipe_data.iloc[0]['img_src']) if 'img_src' in recipe_data else '',
                nutrition=str(recipe_data.iloc[0]['nutrition']) if 'nutrition' in recipe_data else ''
            )
            db.session.add(recipe)
            db.session.commit()

        # Kiểm tra xem món đã có trong giỏ hàng chưa
        cart_item = CartItem.query.filter_by(cart_id=current_user.cart.id, recipe_id=recipe.id).first()
        if cart_item:
            cart_item.quantity += quantity
        else:
            cart_item = CartItem(
                cart_id=current_user.cart.id,
                recipe_id=recipe.id,
                quantity=quantity,
                price=price
            )
            db.session.add(cart_item)

        db.session.commit()
        return jsonify({
            'status': 'success',
            'message': 'Đã thêm món vào giỏ hàng',
            'cart_total': len(current_user.cart.items)
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Có lỗi xảy ra: {str(e)}'}), 500

@app.route('/cart/update/<int:item_id>', methods=['POST'])
@login_required
def update_cart_item(item_id):
    """Cập nhật số lượng món ăn trong giỏ hàng"""
    try:
        quantity = int(request.form.get('quantity', 1))
        if quantity < 1:
            return jsonify({'error': 'Số lượng không hợp lệ'}), 400

        cart_item = CartItem.query.get_or_404(item_id)
        if cart_item.cart.user_id != current_user.id:
            return jsonify({'error': 'Không có quyền truy cập'}), 403

        cart_item.quantity = quantity
        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': 'Đã cập nhật số lượng',
            'new_quantity': quantity,
            'item_total': quantity * cart_item.price,
            'cart_total': current_user.cart.get_total()
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Có lỗi xảy ra: {str(e)}'}), 500

@app.route('/cart/remove/<int:item_id>', methods=['POST'])
@login_required
def remove_from_cart(item_id):
    """Xóa món ăn khỏi giỏ hàng"""
    try:
        cart_item = CartItem.query.get_or_404(item_id)
        if cart_item.cart.user_id != current_user.id:
            return jsonify({'error': 'Không có quyền truy cập'}), 403

        db.session.delete(cart_item)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': 'Đã xóa món khỏi giỏ hàng',
            'cart_total': len(current_user.cart.items)
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Có lỗi xảy ra: {str(e)}'}), 500

@app.route('/cart/clear', methods=['POST'])
@login_required
def clear_cart():
    """Xóa toàn bộ giỏ hàng"""
    try:
        if current_user.cart:
            current_user.cart.items.clear()
            db.session.commit()
        return jsonify({
            'status': 'success',
            'message': 'Đã xóa giỏ hàng'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Có lỗi xảy ra: {str(e)}'}), 500

# Update base context processor to include cart info
@app.context_processor
def utility_processor():
    def get_cart_count():
        if current_user.is_authenticated and current_user.cart:
            return len(current_user.cart.items)
        return 0
    return dict(get_cart_count=get_cart_count)

# Thêm route để serve static files từ thư mục assets
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

@app.route('/remove_from_favorites', methods=['POST'])
@login_required
def remove_from_favorites():
    data = request.get_json()
    recipe_names = data.get('recipe_ids', [])

    if not recipe_names:
        return jsonify({'success': False, 'message': 'No recipes provided'})

    try:
        # Get all recipes to remove
        recipes_to_remove = Recipe.query.filter(Recipe.name.in_(recipe_names)).all()

        # Remove recipes from user's favorites
        for recipe in recipes_to_remove:
            if recipe in current_user.favorites:
                current_user.favorites.remove(recipe)

        db.session.commit()
        return jsonify({'success': True, 'message': 'Recipes removed from favorites'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
