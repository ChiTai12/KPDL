from flask import Blueprint, redirect, url_for, flash, session, request, render_template
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User
import re

auth = Blueprint('auth', __name__)

def validate_password(password):
    """Kiểm tra mật khẩu có đáp ứng yêu cầu:
    - Ít nhất 5 ký tự
    - Có ít nhất 1 ký tự đặc biệt
    - Có ít nhất 1 chữ số
    """
    if len(password) < 5:
        return False, "Mật khẩu phải có ít nhất 5 ký tự"
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Mật khẩu phải chứa ít nhất 1 ký tự đặc biệt"
    
    if not re.search(r"\d", password):
        return False, "Mật khẩu phải chứa ít nhất 1 chữ số"
    
    return True, "Mật khẩu hợp lệ"

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        # Kiểm tra yêu cầu mật khẩu
        is_valid, message = validate_password(password)
        if not is_valid:
            flash(message, 'error')
            return redirect(url_for('auth.login'))

        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            flash('Email hoặc mật khẩu không chính xác', 'error')
            return redirect(url_for('auth.login'))

        login_user(user, remember=remember)
        flash(f'Chào mừng, {user.name}!', 'success')
        return redirect(url_for('home'))

    return render_template('auth/login.html')

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Kiểm tra các trường bắt buộc
        if not email or not name or not password:
            flash('Vui lòng điền đầy đủ thông tin', 'error')
            return redirect(url_for('auth.register'))

        # Kiểm tra mật khẩu xác nhận
        if password != confirm_password:
            flash('Mật khẩu xác nhận không khớp', 'error')
            return redirect(url_for('auth.register'))

        # Kiểm tra yêu cầu mật khẩu
        is_valid, message = validate_password(password)
        if not is_valid:
            flash(message, 'error')
            return redirect(url_for('auth.register'))

        # Kiểm tra email đã tồn tại
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email đã được sử dụng', 'error')
            return redirect(url_for('auth.register'))

        # Tạo user mới
        new_user = User(email=email, name=name)
        new_user.set_password(password)

        # Thêm vào database
        db.session.add(new_user)
        db.session.commit()

        flash('Đăng ký thành công! Vui lòng đăng nhập', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html')

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Đã đăng xuất thành công!', 'success')
    return redirect(url_for('home')) 