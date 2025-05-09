from app import app
from models import User, db

# Mật khẩu mới đáp ứng yêu cầu: ít nhất 5 ký tự, 1 ký tự đặc biệt, 1 chữ số
new_password = "Admin@123"

with app.app_context():
    admin = User.query.filter_by(email="admin@recipe.com").first()
    if admin:
        admin.set_password(new_password)
        db.session.commit()
        print(f"Đã cập nhật mật khẩu cho tài khoản admin: {admin.email}")
        print(f"Mật khẩu mới: {new_password}")
    else:
        print("Không tìm thấy tài khoản admin")
