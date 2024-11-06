import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

# Fonction sigmoide
def sigmoide(solution):
    return 1 / (1 + np.exp(-solution))

def lire_donnees(nom_fichier):
    with open(nom_fichier, 'r') as fichier:
        lignes = fichier.readlines()

    # Poids des objets
    poids = np.array([int(x) for x in lignes[0].strip().split()])

    # Profits des objets
    profits = np.array([int(x) for x in lignes[1].strip().split()])

    # Capacité du sac à dos
    capacite = int(lignes[2])

    return poids, profits, capacite

def fonction_objective(x, profits):
    profit_total = np.sum(profits * x)
    return profit_total

def contraintes(x, poids, capacite):
    poids_total = np.sum(poids * x)
    return poids_total <= capacite

class GWO:
    def __init__(self, fonction_objective, contraintes, args_contraintes, nb_loups, max_iterations):
        self.fonction_objective = fonction_objective
        self.contraintes = contraintes
        self.args_contraintes = args_contraintes
        self.dim = len(self.args_contraintes['poids'])
        self.nb_loups = nb_loups
        self.max_iterations = max_iterations
        self.evolution_fitness = []

    def deux_opt(self, position):
        # Créer une copie de la position initiale
        meilleure_position = position.copy()
        poids_total = np.sum(self.args_contraintes['poids'] * meilleure_position)
        meilleur_profit = self.fonction_objective(meilleure_position, self.args_contraintes['profits'])

        for i in range(len(position)):
            for j in range(i + 1, len(position)):
                if position[i] != position[j]:
                    nouvelle_position = meilleure_position.copy()
                    nouvelle_position[i], nouvelle_position[j] = nouvelle_position[j], nouvelle_position[i]

                    nouveau_poids_total = np.sum(self.args_contraintes['poids'] * nouvelle_position)
                    if nouveau_poids_total <= self.args_contraintes['capacite']:
                        nouveau_profit = self.fonction_objective(nouvelle_position, self.args_contraintes['profits'])
                        if nouveau_profit > meilleur_profit:
                            meilleure_position = nouvelle_position
                            meilleur_profit = nouveau_profit
                            poids_total = nouveau_poids_total

        return meilleure_position

    def initialisation(self):
        position = np.random.uniform(0, 1, (self.nb_loups, self.dim))

        # Calculer le ratio profit sur poids pour chaque objet
        ratios = self.args_contraintes['profits'] / self.args_contraintes['poids']
        objets_tries = np.argsort(ratios)[::-1]

        a = 2

        for t in range(self.max_iterations):
            position, a = self.mise_a_jour_position(position, a, objets_tries)
            a = 2 * (1 - t / self.max_iterations)
            fitness = self.meilleures_solutions(position)[1]
            self.evolution_fitness.append(fitness)

        return self.meilleures_solutions(position)

    def mise_a_jour_position(self, position, a, objets_tries):
        positions_set = {tuple(p) for p in position}

        for i in range(self.nb_loups):
            if len(objets_tries) < 3:
                break

            fitness = [self.fonction_objective(position[i], self.args_contraintes['profits']) for i in range(self.nb_loups)]
            trier = np.argsort(fitness)[::-1]

            X_alpha = position[trier[0]].copy()
            X_beta = position[trier[1]].copy()
            X_gamma = position[trier[2]].copy()

            A = 2 * a * np.random.rand() - a
            C = 2 * np.random.rand() - 1
            D_alpha = np.abs(C * X_alpha - position[i])
            D_beta = np.abs(C * X_beta - position[i])
            D_gamma = np.abs(C * X_gamma - position[i])
            X1 = X_alpha - A * D_alpha
            X2 = X_beta - A * D_beta
            X3 = X_gamma - A * D_gamma
            new_position = (X1 + X2 + X3) / 3

            # Appliquer la fonction sigmoide
            new_position = sigmoide(new_position)

            # Mettre à jour les valeurs de position à 1 si >= 0.5, sinon à 0
            new_position = np.where(new_position >= 0.5, 1, 0)

            # Vérifier les contraintes
            poids_total = np.sum(self.args_contraintes['poids'] * new_position)
            while poids_total > self.args_contraintes['capacite']:
                for j in objets_tries[::-1]:
                    if new_position[j] == 1:
                        new_position[j] = 0
                        poids_total -= self.args_contraintes['poids'][j]
                    if poids_total <= self.args_contraintes['capacite']:
                        break

            # Vérifier l'unicité de la position
            while tuple(new_position) in positions_set:
                new_position = np.random.randint(2, size=self.dim)
                poids_total = np.sum(self.args_contraintes['poids'] * new_position)
                while poids_total > self.args_contraintes['capacite']:
                    for j in objets_tries[::-1]:
                        if new_position[j] == 1:
                            new_position[j] = 0
                            poids_total -= self.args_contraintes['poids'][j]
                        if poids_total <= self.args_contraintes['capacite']:
                            break

            positions_set.add(tuple(new_position))
            position[i] = new_position
        return position, a

    def meilleures_solutions(self, position):
        fitness = [self.fonction_objective(position[i], self.args_contraintes['profits']) for i in range(self.nb_loups)]
        trier = np.argsort(fitness)[::-1]
        X_alpha = position[trier[0]].copy()
        alpha_fitness = fitness[trier[0]]
        X_beta = position[trier[1]].copy()
        beta_fitness = fitness[trier[1]]
        X_gamma = position[trier[2]].copy()
        gamma_fitness = fitness[trier[2]]
        poids_total_alpha = np.sum(self.args_contraintes['poids'] * X_alpha)
        poids_total_beta = np.sum(self.args_contraintes['poids'] * X_beta)
        poids_total_gamma = np.sum(self.args_contraintes['poids'] * X_gamma)
        return X_alpha, alpha_fitness, poids_total_alpha, X_beta, beta_fitness, poids_total_beta, X_gamma, gamma_fitness, poids_total_gamma

def charger_donnees():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    chemin_fichier = filedialog.askopenfilename(title="Sélectionner le fichier de données")
    return chemin_fichier

# Charger les données depuis le fichier sélectionné
fichier = charger_donnees()
if fichier:
    poids, profits, capacite = lire_donnees(fichier)

    # Configuration de l'algorithme
    gwo = GWO(fonction_objective, contraintes, {'poids': poids, 'profits': profits, 'capacite': capacite}, nb_loups=10, max_iterations=200)
    position_alpha, fitness_alpha, poids_total_alpha, position_beta, fitness_beta, poids_total_beta, position_gamma, fitness_gamma, poids_total_gamma = gwo.initialisation()

    print("Meilleure solution X_alpha:")
    print("Position:", position_alpha)
    print("Fitness:", fitness_alpha)
    print("Poids total:", poids_total_alpha)
    print()
    print("Deuxieme meilleure solution X_beta:")
    print("Position:", position_beta)
    print("Fitness:", fitness_beta)
    print("Poids total:", poids_total_beta)
    print()
    print("Troisieme meilleure solution X_gamma:")
    print("Position:", position_gamma)
    print("Fitness:", fitness_gamma)
    print("Poids total:", poids_total_gamma)

    # Afficher le graphique de l'évolution de la fitness
    plt.plot(gwo.evolution_fitness)
    plt.title("Évolution de la fitness de la meilleure solution (X_alpha)")
    plt.xlabel("Itération")
    plt.ylabel("Fitness")
    plt.show()
