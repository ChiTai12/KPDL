from app import app, User, db

with app.app_context():
    # Xóa tất cả user không phải admin
    users = User.query.filter_by(is_admin=False).all()
    for user in users:
        print(f"Đang xóa user: {user.email}")
        db.session.delete(user)
    db.session.commit()
    print("Đã xóa xong các tài khoản không phải admin") 