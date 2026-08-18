import math
import random
import shutil
import json
from sklearn.datasets import load_digits


def centrer(phrase, delimiteur=""):
    l = shutil.get_terminal_size().columns
    if delimiteur == "":
        print(f"{phrase}".center(l))
    else:
        print(f"{phrase}".center(l, delimiteur))


# 1. Chargement et préparation des données
digit = load_digits()
x = digit.images
y = digit.target

# Transformation des images 8x8 en listes de 64 pixels (normalisés entre 0 et 1)
x_flat = []
for img in x:
    ligne_pixel = [pixel / 16.0 for ligne in img for pixel in ligne]
    x_flat.append(ligne_pixel)

# Encodage One-Hot des cibles (10 sorties)
y_flat = []
for a in y:
    cible = [0] * 10
    cible[a] = 1
    y_flat.append(cible)


class RESEAU_NEURONE:

    def __init__(self, taille_entre, taille_cache, taille_sortie):
        self.taille_entre = taille_entre
        self.taille_cache = taille_cache
        self.taille_sortie = taille_sortie

        # Initialisation des poids
        self.poid_cache = [
            [random.uniform(-1, 1) for _ in range(taille_entre)]
            for _ in range(taille_cache)
        ]
        self.biais_cache = [random.uniform(-1, 1) for _ in range(taille_cache)]

        self.poid_sortie = [
            [random.uniform(-1, 1) for _ in range(taille_cache)]
            for _ in range(taille_sortie)
        ]
        self.biais_sortie = [
            random.uniform(-1, 1) for _ in range(taille_sortie)
        ]
    def sauvegarder(self):
        return{
            "taille_entre": self.taille_entre,
            "taille_cache": self.taille_cache,
            "taille_sortie": self.taille_sortie,
            "poid_cache": self.poid_cache,
            "biais_cache": self.biais_cache,
            "poid_sortie": self.poid_sortie,
            "biais_sortie": self.biais_sortie
        }
    def charger(self, donnees):
        self.taille_entre = donnees["taille_entre"]
        self.taille_cache = donnees["taille_cache"]
        self.taille_sortie = donnees["taille_sortie"]
        self.poid_cache = donnees["poid_cache"]
        self.biais_cache = donnees["biais_cache"]
        self.poid_sortie = donnees["poid_sortie"]
        self.biais_sortie = donnees["biais_sortie"]

    def sigmoide(self, x):
        return 1 / (1 + math.exp(-x))

    def deriv_sigmoide(self, x):
        return x * (1 - x)

    def forward(self, x_flat):
        self.sortie_cache = []
        for a in range(self.taille_cache):
            score = 0
            for b in range(self.taille_entre):
                score += x_flat[b] * self.poid_cache[a][b]
            score += self.biais_cache[a]
            self.sortie_cache.append(self.sigmoide(score))

        self.prediction = []
        for c in range(self.taille_sortie):
            score_sortie = 0
            for d in range(self.taille_cache):
                score_sortie += (
                    self.sortie_cache[d] * self.poid_sortie[c][d]
                )
            score_sortie += self.biais_sortie[c]
            self.prediction.append(self.sigmoide(score_sortie))

        return self.prediction

    def backward(self, x_flat, cible, lr):
        preds = self.forward(x_flat)

        # 1. Calcul des gradients de sortie
        gradient_sortie = []
        for a in range(self.taille_sortie):
            erreur = preds[a] - cible[a]
            grad = erreur * self.deriv_sigmoide(preds[a])
            gradient_sortie.append(grad)

        # 2. Calcul des gradients de la couche cachée
        gradient_cache = []
        for b in range(self.taille_cache):
            erreur_couche_cachee = sum(
                gradient_sortie[c] * self.poid_sortie[c][b]
                for c in range(self.taille_sortie)
            )
            grad = erreur_couche_cachee * self.deriv_sigmoide(
                self.sortie_cache[b]
            )
            gradient_cache.append(grad)

        # 3. Mise à jour des poids et biais de sortie
        for c in range(self.taille_sortie):
            for d in range(self.taille_cache):
                self.poid_sortie[c][d] -= (
                    lr * gradient_sortie[c] * self.sortie_cache[d]
                )
            self.biais_sortie[c] -= lr * gradient_sortie[c]

        # 4. Mise à jour des poids et biais de la couche cachée
        for e in range(self.taille_cache):
            for f in range(self.taille_entre):
                self.poid_cache[e][f] -= (
                    lr * gradient_cache[e] * x_flat[f]
                )
            self.biais_cache[e] -= lr * gradient_cache[e]

    def train(self, epoques, x_flat, y_flat, lr):
        for epoque in range(epoques):
            for i in range(len(x_flat)):
                self.backward(x_flat[i], y_flat[i], lr)

        
            if epoque % 20 == 0 :
           
                prediction_derniere_img = self.forward(x_flat[i])
                cible_derniere_img = y_flat[i]

                print("\n")
                centrer(f"--- Bilan Époque {epoque} (Dernière image) ---\n")
                for r, u in enumerate(prediction_derniere_img):
                    ecart = u - cible_derniere_img[r]
                    centrer(
                        f"Neurone {r} | Prédit : {u:.4f} | Cible : {cible_derniere_img[r]} | Écart : {ecart:.4f}"
                    )
            


if __name__ == "__main__":
    # La taille d'entrée est de 64 (pixels) et la taille de sortie est de 10 (chiffres de 0 à 9)
    CHIFFRE = RESEAU_NEURONE(
        taille_entre=64, taille_cache=16, taille_sortie=10
    )

    centrer("DEBUT DE L'ENTRAINEMENT", "-")
    CHIFFRE.train(epoques=100, lr=0.2, x_flat=x_flat, y_flat=y_flat)

    centrer("FIN DE L'ENTRAINEMENT", "-")

    centrer("chargement du cerveau")

    cerveau= CHIFFRE.sauvegarder()

    with open("load_cerveau_4.json", "w") as f:
        json.dump(cerveau, f, indent= 4)

    

