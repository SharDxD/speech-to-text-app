from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["speech_app"]

users = db["users"]
transcripts = db["transcripts"]
