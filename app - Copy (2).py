from flask import Flask, request, render_template, session, redirect
import joblib
import numpy as np
import os
import contextlib
import re
import sqlite3
import pandas as pd
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from create_database import setup_database
from utils import login_required, set_session

app = Flask(__name__)

# ✅ Load the model and feature names
model, feature_names = joblib.load('heart_disease_model.pkl')

# ✅ Set up the user database
database = "users.db"
setup_database(name=database)

# ✅ Set the Flask secret key
app.secret_key = 'xpSm7p5bgJY8rNoBjGWiz5yjxM-NEBlW6SIBI62OkLc='

# ✅ Home Route
@app.route('/')
def home():
    return render_template('index.html')

# ✅ About Route
@app.route('/about')
def about():
    return render_template('about.html')

# ✅ Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    # Get input from form
    username = request.form.get('username')
    password = request.form.get('password')

    # Query the database for user info
    query = 'SELECT username, password, email FROM users WHERE username = :username'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            account = conn.execute(query, {'username': username}).fetchone()

    if not account:
        return render_template('login.html', error='Username does not exist')

    # Verify password
    try:
        ph = PasswordHasher()
        ph.verify(account[1], password)
    except VerifyMismatchError:
        return render_template('login.html', error='Incorrect password')

    # Update password hash if needed
    if ph.check_needs_rehash(account[1]):
        query = 'UPDATE users SET password = :password WHERE username = :username'
        params = {'password': ph.hash(password), 'username': account[0]}
        with contextlib.closing(sqlite3.connect(database)) as conn:
            with conn:
                conn.execute(query, params)

    # Set session
    set_session(username=account[0], email=account[2], remember_me='remember-me' in request.form)
    return redirect('/predict_page')

# ✅ Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    username = request.form.get('username')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm-password')
    email = request.form.get('email')

    # ✅ Validate input
    if len(password) < 8:
        return render_template('register.html', error='Password must be at least 8 characters')
    if password != confirm_password:
        return render_template('register.html', error='Passwords do not match')
    if not re.match(r'^[a-zA-Z0-9]+$', username):
        return render_template('register.html', error='Username must only contain letters and numbers')
    if not 3 < len(username) < 26:
        return render_template('register.html', error='Username must be between 4 and 25 characters')

    # ✅ Check if username already exists
    query = 'SELECT username FROM users WHERE username = :username'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, {'username': username}).fetchone()

    if result:
        return render_template('register.html', error='Username already exists')

    # ✅ Create password hash
    ph = PasswordHasher()
    hashed_password = ph.hash(password)

    # ✅ Insert into database
    query = 'INSERT INTO users (username, password, email) VALUES (:username, :password, :email)'
    params = {'username': username, 'password': hashed_password, 'email': email}
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            conn.execute(query, params)

    # ✅ Set session
    set_session(username=username, email=email)
    return redirect('/')

# ✅ Predict Page Route
@app.route('/predict_page', methods=['GET'])
def predict_page():
    return render_template('predict.html')

# ✅ Prediction Route
@app.route('/result', methods=['POST'])
def result():
    try:
        # ✅ Get input values and convert to float
        input_data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
    except ValueError:
        return render_template('predict.html', error="Invalid input. Please enter valid numeric values.")

    # ✅ Convert input data to a dataframe and match feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # ✅ Make prediction
    prediction = model.predict(input_df)[0]
    result = "Your data is safe" if prediction == 1 else "Your data is not safe"

    return render_template('result.html', prediction=result)

# ✅ Logout Route
@app.route('/logout')
def logout():
    session.clear()
    session.permanent = False
    return redirect('/')

# ✅ Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
