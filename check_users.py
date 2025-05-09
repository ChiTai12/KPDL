from app import app, User

with app.app_context():
    users = User.query.all()
    print("\nDanh sách tài khoản:")
    for user in users:
        print(f"- Email: {user.email}")
        print(f"  Tên: {user.name}")
        print(f"  Admin: {user.is_admin}")
        print() 