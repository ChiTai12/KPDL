from flask import Flask
from models import db

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recipe_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def remove_comments_table():
    with app.app_context():
        # Xóa bảng comments nếu tồn tại
        db.engine.execute('DROP TABLE IF EXISTS comment')
        print("Đã xóa bảng comments thành công!")

if __name__ == '__main__':
    remove_comments_table() 