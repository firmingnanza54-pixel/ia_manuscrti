let pixels = new Array(64).fill(0.0);

document.addEventListener('DOMContentLoaded', function(){
    const grille = document.getElementById('grille');

    for (let i = 0; i < 64; i++){
        const divpixel = document.createElement('div');
        divpixel.classList.add('pixel');
        divpixel.addEventListener('click', function(){
            if (pixels[i] === 0.0){
                pixels[i] = 1.0;
                divpixel.classList.add('actif');
            } else {
                pixels[i] = 0.0;
                divpixel.classList.remove('actif');
            }
        });

        grille.appendChild(divpixel);
    }
});

// Placé en dehors pour être accessible par onclick=""
function effacer_grille(){
    pixels.fill(0.0);

    const cases = document.querySelectorAll('.pixel');
    cases.forEach(function(unecase){
        unecase.classList.remove('actif');
    });

    document.getElementById('resultat').innerText = "Dessinez un chiffre puis appuyez sur le bouton prédire";
}

function envoyer_prediction() {
    document.getElementById('resultat').innerText = "Calcul en cours ...";

    fetch('/predire', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 'image': pixels })
    })
    .then(reponse => reponse.json()) 
    .then(data => {
        document.getElementById('resultat').innerText = "Résultat prédit : " + data.pred + " (certitude : " + data.certitude + "%)";
    })
    .catch(error => {
        console.error("Erreur :", error);
        document.getElementById('resultat').innerText = "Une erreur est survenue pendant la prédiction";
    });
}