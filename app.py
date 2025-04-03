from flask import Flask, jsonify, request, send_file, render_template,send_from_directory,url_for, flash, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length
import requests
import cv2
import os
import numpy as np

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

UPLOAD_FOLDER = 'templates/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

class RegistrationForm(FlaskForm):
    username = StringField('username', validators=[DataRequired(), Length(min=2, max=20)])
    password = PasswordField('password', validators=[DataRequired()])
    submit = SubmitField('Sign Up')

def fetch_user_by_username(username):
    return User.query.filter_by(username=username).first()

def fetch_user_by_id(user_id):
    return User.query.get(int(user_id))

def create_user(username, password, is_admin=False):
    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()


def fetch_all_users():
    return User.query.all()


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('recognize'))
    return redirect(url_for('login'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = fetch_user_by_username(username)

        if user and user.password == password:
            login_user(user)
            return redirect(url_for('recognize'))
        else:
            flash('Invalid username or password. Please try again.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegistrationForm()

    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        existing_user = fetch_user_by_username(username)
        if existing_user:
            flash('Username already exists. Please choose a different username.', 'error')
        else:
            create_user(username, password)
            flash('Account created successfully. Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html', form=form)

@app.route('/recognize')
@login_required
def recognize():
    return render_template('recognize.html')

# Define a dictionary to store dataset information
datasets = {
    'plantvillage': {
        'name': 'PlantVillage Dataset',
        'description': 'Dataset containing images of healthy and diseased plant leaves from various crops.',
        'url': 'https://plantvillage.psu.edu/data'
    }}
# Define routes

def generate_description(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image (e.g., resize, normalize)

    # Use a pre-trained machine learning model to classify the disease
    # Replace this with your actual model loading and inference code
    # For demonstration purposes, we'll use a placeholder result
    disease_class = "Placeholder Disease"
    confidence = 0.75



@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        return render_template('recognize.html', error='No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('recognize.html', error='No file selected')

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("File path:", file_path)  # Print out the file path for debugging
        file.save(file_path)  # Save the uploaded file to the uploads folder
        # Preprocess the uploaded image
        image = preprocess_image(file_path)
        # Perform disease recognition
        prediction = recognize_disease(file_path)
        return render_template('recognize.html', filename=filename, prediction=prediction)
# Helper functions

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Perform preprocessing steps (resize, normalize, etc.)
    # Here, we'll just resize the image to a fixed size (224x224)
    image = cv2.resize(image, (224, 224))
    return image

def recognize_disease(image_path):  # Accept image_path argument
    image = cv2.imread(image_path)
    # Load the machine learning model for disease recognition
    model = load_model()
    # Perform disease recognition
    # Here, we'll use a placeholder implementation using OpenCV's CascadeClassifier
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        return {'disease': 'Disease Detected', 'confidence': 1.0}
    else:
        return {'disease': 'No Disease Detected', 'confidence': 1.0}

def load_model():
    # Placeholder function for loading the trained model
    # Replace this with your actual model loading code
    return None

# API routes

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    # Placeholder function for extracting dataset
    # Replace this with your actual dataset extraction code
    dataset = extract_dataset()
    return jsonify(dataset)

def extract_dataset():
    # Placeholder function for extracting dataset
    # Replace this with your actual dataset extraction code
    # For demonstration purposes, we'll return a sample dataset
    return {'dataset_name': 'Plant Disease Dataset', 'num_images': 1000}

# Route to list available datasets
@app.route('/datasets', methods=['GET'])
def list_datasets():
    return jsonify(datasets)

# Route to download images from a dataset
@app.route('/download', methods=['GET'])
def download_image():
    dataset = request.args.get('dataset')
    image_url = request.args.get('image_url')

    if dataset not in datasets:
        return jsonify({'error': 'Dataset not found'}), 404

    # Download the image from the specified URL
    image_response = requests.get(image_url)
    if image_response.status_code != 200:
        return jsonify({'error': 'Failed to download image'}), 500

    # Save the image to a temporary file
    with open('temp_image.jpg', 'wb') as f:
        f.write(image_response.content)

    # Send the image file to the client
    return send_file('temp_image.jpg', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
