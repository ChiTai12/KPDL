from app import app
from models import User

with app.app_context():
    admin_users = User.query.filter_by(is_admin=True).all()
    print("Admin users:")
    for user in admin_users:
        print(f"Email: {user.email}, Name: {user.name}")
    
    # Kiểm tra tất cả người dùng
    all_users = User.query.all()
    print("\nAll users:")
    for user in all_users:
        print(f"Email: {user.email}, Name: {user.name}, Is Admin: {user.is_admin}")
