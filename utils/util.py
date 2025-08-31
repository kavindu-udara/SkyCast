from datetime import datetime
import uuid
from bson import ObjectId

def generate_random_name():
    return datetime.now().strftime('%Y%m%d_%H%M%S_') + str(uuid.uuid4())

def convert_objectid_to_str(doc):
    if isinstance(doc, list):
        return [convert_objectid_to_str(d) for d in doc]
    if isinstance(doc, dict):
        return {k: str(v) if isinstance(v, ObjectId) else convert_objectid_to_str(v) for k, v in doc.items()}
    return doc
