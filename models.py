from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200))
    name = db.Column(db.String(100), nullable=False)
    picture = db.Column(db.String(200), default='static/default_avatar.png')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    google_id = db.Column(db.String(100), unique=True)
    is_admin = db.Column(db.Boolean, default=False)
    
    # Relationships
    favorites = db.relationship('Recipe', secondary='user_favorites', backref='favorited_by', cascade='all, delete')
    recipes = db.relationship('Recipe', backref='author', lazy=True, cascade='all, delete-orphan')
    ratings = db.relationship('Rating', backref='author', lazy=True, cascade='all, delete-orphan')
    cart = db.relationship('Cart', backref='user', lazy=True, uselist=False, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Recipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, unique=True)
    ingredients = db.Column(db.Text)
    instructions = db.Column(db.Text)
    image_url = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'))
    rating = db.Column(db.Float, default=0.0)
    prep_time = db.Column(db.String(50))
    cook_time = db.Column(db.String(50))
    total_time = db.Column(db.String(50))
    img_src = db.Column(db.String(500))
    nutrition = db.Column(db.Text)
    
    # Relationships
    ratings = db.relationship('Rating', backref='recipe', lazy=True, cascade='all, delete-orphan')

    def __init__(self, **kwargs):
        # Convert ingredients to string if it's a list
        if 'ingredients' in kwargs and isinstance(kwargs['ingredients'], (list, tuple)):
            kwargs['ingredients'] = ', '.join(map(str, kwargs['ingredients']))
            
        # Ensure rating is float
        if 'rating' in kwargs:
            try:
                kwargs['rating'] = float(kwargs['rating'])
            except (ValueError, TypeError):
                kwargs['rating'] = 0.0
                
        # Ensure other fields are strings
        for field in ['name', 'prep_time', 'cook_time', 'total_time', 'img_src', 'nutrition']:
            if field in kwargs:
                kwargs[field] = str(kwargs[field])
                
        super(Recipe, self).__init__(**kwargs)

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    recipe_id = db.Column(db.Integer, db.ForeignKey('recipe.id', ondelete='CASCADE'), nullable=False)

# Association table for user favorites
user_favorites = db.Table('user_favorites',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), primary_key=True),
    db.Column('recipe_id', db.Integer, db.ForeignKey('recipe.id', ondelete='CASCADE'), primary_key=True)
)

class Cart(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    items = db.relationship('CartItem', backref='cart', lazy=True, cascade='all, delete-orphan')
    
    def get_total(self):
        return sum(item.quantity * item.price for item in self.items)

class CartItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cart_id = db.Column(db.Integer, db.ForeignKey('cart.id', ondelete='CASCADE'), nullable=False)
    recipe_id = db.Column(db.Integer, db.ForeignKey('recipe.id', ondelete='CASCADE'), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    price = db.Column(db.Float, default=0.0)
    
    recipe = db.relationship('Recipe', backref='cart_items') 