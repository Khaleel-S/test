import json
import uuid
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import pandas as pd
import logging
import joblib
import spacy
import re
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from datetime import datetime
import os

# --- Configuration and Setup ---
logging.basicConfig(level=logging.INFO)
analysis_sessions = {} 

# Placeholder for NLP resources
model = None
vectorizer = None
nlp = None
stop_words = set()
MODEL_READY = False

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

try:
    # NOTE: Ensure sentiment_model.pkl and vectorizer.pkl are in the same directory.
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    
    # NOTE: Ensure 'en_core_web_sm' is downloaded (python -m spacy download en_core_web_sm)
    nlp = spacy.load("en_core_web_sm")
    
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    logging.info("ML Model, Vectorizer, and SpaCy loaded successfully.")
    MODEL_READY = True
except Exception as e:
    logging.error(f"Error loading NLP resources: {e}")

# --- Core Logic Functions ---

def preprocess(text):
    """Simple cleaning function for the ML model."""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return ' '.join([word for word in text.split() if word not in stop_words])

def analyze_review_sentiment(review_text):
    """Analyzes overall sentiment using the trained ML model."""
    if not MODEL_READY or not review_text.strip():
        return {"sentiment": "Neutral", "confidence": 0.0}

    processed_text = preprocess(review_text)
    
    try:
        X_vec = vectorizer.transform([processed_text])
        sentiment = model.predict(X_vec)[0]
        probabilities = model.predict_proba(X_vec)[0]
        confidence = max(probabilities)
    except Exception:
        return {"sentiment": "Neutral", "confidence": 0.5}

    return {"sentiment": sentiment, "confidence": float(f"{confidence:.3f}")}

def aspect_based_sentiment(review_text):
    """Performs basic Aspect-Based Sentiment Analysis using SpaCy."""
    if not nlp: return {}
    overall_sentiment = analyze_review_sentiment(review_text)['sentiment']
    doc = nlp(review_text.lower())
    aspects = {}
    
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) < 4:
            aspect = chunk.text.strip()
            aspects[aspect] = overall_sentiment
    
    common_junk = ['product', 'item', 'thing', 'review', 'i', 'service', 'day', 'time']
    return {k: v for k, v in aspects.items() if k not in common_junk and len(k) > 2}

def get_recommendation(summary):
    """Generates a simple 'Good' or 'Bad' purchase recommendation based on positive vs. negative ratio."""
    positive = summary.get("Positive", 0)
    negative = summary.get("Negative", 0)
    total_sentiment = positive + negative
    
    if total_sentiment < 5: 
        return {"status": "Neutral", "message": "Not enough strong sentiment to make a reliable recommendation."}

    positivity_ratio = positive / total_sentiment
    
    if positivity_ratio >= 0.70:
        return {"status": "Good", "message": "The product has overwhelmingly positive feedback."}
    elif positivity_ratio >= 0.55:
        return {"status": "Decent", "message": "The product has slightly more positive than negative feedback."}
    else:
        return {"status": "Bad", "message": "The product has significant or dominant negative feedback."}

# --- Flask Application Setup (Routes) ---

app = Flask(__name__)
CORS(app) 

@app.route('/')
def index():
    """Renders the main dashboard page, handling optional session ID loading."""
    session_id = request.args.get('session_id')
    return render_template('index.html', session_id=session_id)

@app.route('/api/analyze_and_redirect', methods=['POST'])
def analyze_and_redirect():
    """Receives data, processes it, stores results, and returns a redirect URL."""
    if not MODEL_READY:
        return jsonify({"error": "ML Model not loaded. Check server logs."}), 503
        
    try:
        data = request.get_json()
        reviews_list = data.get('reviews', [])
        
        if not reviews_list:
            return jsonify({"error": "No reviews received for analysis."}), 400

        results = []
        aspect_matrix = defaultdict(lambda: defaultdict(int))
        current_date_str = datetime.now().strftime('%Y-%m-%d')
        
        for review in reviews_list:
            if not review.strip(): continue
            overall = analyze_review_sentiment(review)
            aspects = aspect_based_sentiment(review)
            
            for aspect, sentiment in aspects.items():
                aspect_matrix[aspect][sentiment] += 1
            
            results.append({
                "review": review,
                "sentiment": overall['sentiment'],
                "confidence": overall['confidence'],
                "aspects": aspects,
                "rating": None, 
                "date": current_date_str
            })
        
        if not results:
             return jsonify({"error": "No valid reviews were received for analysis."}), 400

        df = pd.DataFrame(results)
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        
        summary = {
            "total": len(results),
            "Positive": sentiment_counts.get("Positive", 0),
            "Negative": sentiment_counts.get("Negative", 0),
            "Neutral": sentiment_counts.get("Neutral", 0)
        }
        
        drill_down_matrix = []
        for aspect, counts in aspect_matrix.items():
            drill_down_matrix.append({
                "aspect": aspect,
                "Positive": counts.get("Positive", 0),
                "Negative": counts.get("Negative", 0),
                "Neutral": counts.get("Neutral", 0),
                "total": sum(counts.values())
            })
        drill_down_matrix.sort(key=lambda x: x['total'], reverse=True)
        
        # --- Store Product Data ---
        recommendation = get_recommendation(summary)
        product_name = data.get('product_name', "Analyzed Product")
        product_image_url = data.get('product_image_url', "") 
        
        session_id = str(uuid.uuid4())
        analysis_sessions[session_id] = {
            "product_name": product_name,
            "product_image_url": product_image_url, 
            "recommendation": recommendation,
            "summary": summary,
            "detailed_results": results,
            "drill_down_matrix": drill_down_matrix
        }
        
        logging.info(f"Analysis complete. Returning redirect URL for session_id: {session_id}")
        
        redirect_url = url_for('index', session_id=session_id, _external=True)

        return jsonify({
            "status": "success",
            "redirect_url": redirect_url
        }), 200

    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}", exc_info=True)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/api/get_session_data/<session_id>', methods=['GET'])
def get_session_data(session_id):
    """Retrieves stored analysis results for the dashboard to display."""
    data = analysis_sessions.get(session_id)
    
    if data:
        return jsonify({
            "type": "fetched_bulk",
            "product_name": data.get('product_name', 'Analyzed Product'),
            "product_image_url": data.get('product_image_url', ''), 
            "recommendation": data.get('recommendation', {}),
            "summary": data['summary'],
            "detailed_results": data['detailed_results'],
            "drill_down_matrix": data['drill_down_matrix']
        })
    else:
        return jsonify({"error": "Session data not found or expired."}), 404

if __name__ == '__main__':
    print("---------------------------------------------------------")
    print("      Starting Flask Server - DO NOT CLOSE THIS WINDOW      ")
    print("      Dashboard URL: http://127.0.0.1:5000/               ")
    print("---------------------------------------------------------")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    