import json
from flask import Flask, jsonify, request, render_template
from manuscrit import RESEAU_NEURONE

app = Flask(__name__)

ia = RESEAU_NEURONE(taille_entre=64, taille_cache=16, taille_sortie=10)

with open("load_cerveau_4.json", "r") as f:
    cerveau = json.load(f)

ia.charger(cerveau)

@app.route('/')
def acceuil():
    return render_template('index.html')

@app.route("/predire", methods=["POST"])
def predire():
    donnees = request.get_json()
    pixels = donnees['image']  

    prediction = ia.forward(pixels)

    resultat = prediction.index(max(prediction))
    certitude = round(max(prediction) * 100, 2)

    
    return jsonify({"pred": resultat, "certitude": certitude})

if __name__ == "__main__":
    app.run(debug=True, port=5000)