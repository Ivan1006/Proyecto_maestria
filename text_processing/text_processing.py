import re
from unicodedata import normalize

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text
