from motor.motor_asyncio import AsyncIOMotorClient
import os
import ssl
from dotenv import load_dotenv
from pymongo import MongoClient
from urllib.parse import quote_plus

load_dotenv()

# Load and encode credentials
username = quote_plus(os.getenv("MONGODB_USERNAME", ""))
password = os.getenv("MONGODB_PASSWORD", "")
cluster = os.getenv("MONGODB_CLUSTER")
app_name = os.getenv("MONGODB_APP_NAME", "")

# Construct URI
uri = f"mongodb+srv://{username}:{password}@{cluster}/{app_name}?retryWrites=true&w=majority&authSource=admin"

def connect_db():
    client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=True)
    return client[app_name]
