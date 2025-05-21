# auth.py
from flask import Blueprint, request, redirect, session, render_template, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from db import users

auth = Blueprint("auth", __name__)

@auth.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        role = request.form.get("role", "client")

        if users.find_one({"email": email}):
            return render_template("register.html", error="User already exists")


        hashed_pw = generate_password_hash(password)
        users.insert_one({
            "email": email,
            "password": hashed_pw,
            "role": role
        })
        return redirect(url_for("auth.login"))

    return render_template("register.html")

@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = users.find_one({"email": email})
        if not user or not check_password_hash(user["password"], password):
            return render_template("login.html", error="Invalid credentials")

        session["user_id"] = str(user["_id"])
        session["email"] = user["email"]
        session["role"] = user["role"]
        return redirect(url_for("index"))

    return render_template("login.html")

@auth.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))

