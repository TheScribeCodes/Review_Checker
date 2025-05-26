
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import threading
import webbrowser

# Sample data and model training
data = {
    "review": [
        "This product changed my life! Best purchase ever!!!",
        "Fast delivery, works as expected. Would buy again.",
        "Amazing quality. Highly recommend it to everyone!",
        "Buy this now! Incredible product! So happy!",
        "The item arrived on time and was as described.",
        "Fake product. Totally useless. Don’t waste your money.",
        "I received a different product than advertised. Not happy.",
        "This is a scam. It broke after one use.",
        "Five stars! Best best best! Buy it now!",
        "Excellent value. Packaging was secure and product was intact."
    ],
    "label": [1, 0, 0, 1, 0, 1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["review"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<title>Review Authenticity Checker</title>
<h2>Enter a product review or URL to check for fake reviews</h2>
<form method=post>
  <textarea name=review rows=3 cols=60 placeholder="Paste a review or product URL here..."></textarea><br><br>
  <input type=submit value=Check>
</form>
{% if results %}
  <h3>Results:</h3>
  <ul>
    {% for review, score in results %}
      <li><strong>{{ score }}%</strong> fake: {{ review }}</li>
    {% endfor %}
  </ul>
{% endif %}
"""

def scrape_amazon_reviews(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        review_blocks = soup.select('.review-text-content span')
        reviews = [r.get_text(strip=True) for r in review_blocks]
        return reviews[:5]
    except Exception as e:
        return [f"Error scraping URL: {str(e)}"]

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    if request.method == "POST":
        input_text = request.form["review"]
        results = []
        if input_text.startswith("http"):
            reviews = scrape_amazon_reviews(input_text)
        else:
            reviews = [input_text]
        for rev in reviews:
            X_new = vectorizer.transform([rev])
            prob_fake = model.predict_proba(X_new)[0][1]
            results.append((rev, round(prob_fake * 100, 2)))
    return render_template_string(HTML_TEMPLATE, results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
