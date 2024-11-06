import numpy as np

def generer_fichier_donnees(chemin_fichier):
    nombre_de_variable = 10000
    nombre_de_classe = 5
    capacite = 126900  # Capacité proportionnelle au nombre de variables
    
    # Générer aléatoirement la répartition des variables entre les classes
    repartition = np.random.randint(1, 2000, size=nombre_de_classe)
    repartition = repartition * (nombre_de_variable // np.sum(repartition))
    repartition[-1] += nombre_de_variable - np.sum(repartition)
    
    # Générer aléatoirement les profits et les poids pour chaque variable
    profits = np.random.randint(1, 5000, size=nombre_de_variable)
    poids = np.random.randint(1, 100, size=nombre_de_variable)
    
    # Écrire les données dans le fichier
    with open(chemin_fichier, 'w') as f:
        f.write(f"{nombre_de_variable}\n")
        f.write(f"{nombre_de_classe}\n")
        f.write(f"{capacite}\n")
        f.write(' '.join(map(str, repartition)) + '\n')
        for i in range(nombre_de_variable):
            f.write(f"{profits[i]} {poids[i]}\n")

    print(f"Le fichier de données a été généré avec succès : {chemin_fichier}")

# Exemple d'utilisation
chemin_fichier = "donnees_test.txt"
generer_fichier_donnees(chemin_fichier)
