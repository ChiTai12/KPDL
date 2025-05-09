from app import app, User, db

with app.app_context():
    # Tìm tài khoản admin
    admin = User.query.filter_by(email="admin@recipe.com").first()
    if admin:
        # Cập nhật mật khẩu mới: admin@123 (chứa ký tự đặc biệt @ và số 123)
        admin.set_password("admin@123")
        db.session.commit()
        print("Đã cập nhật mật khẩu admin thành: admin@123")
    else:
        print("Không tìm thấy tài khoản admin") 