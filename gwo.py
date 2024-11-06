import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

# Fonction sigmoide
def sigmoide(solution):
    return 1 / (1 + np.exp(-solution))

def lire_donnees(fichier):
    with open(fichier, 'r') as f:
        # Lire les premières lignes pour extraire les paramètres
        nombre_de_variable = int(f.readline().strip())
        nombre_de_classe = int(f.readline().strip())
        capacite = int(f.readline().strip())
        
        # Lire la répartition des variables
        repartition = list(map(int, f.readline().strip().split()))
        
        # Lire les colonnes des profits et des poids
        profits = []
        poids = []
        for _ in range(nombre_de_variable):
            profit, poid = map(int, f.readline().strip().split())
            profits.append(profit)
            poids.append(poid)
            
    return nombre_de_variable, nombre_de_classe, capacite, repartition, np.array(profits), np.array(poids)

def fonction_objective(x, profits, repartition, nombre_de_variable):
    classe_profits = []
    start = 0
    for classe in repartition:
        end = start + classe
        classe_indices = np.arange(start, end)
        classe_profit = np.sum(profits[classe_indices] * x[classe_indices])
        classe_profits.append(classe_profit)
        start = end
    return np.min(classe_profits)

def contraintes(x, args_contraintes):
    poids_total = np.sum(args_contraintes['poids'] * x)
    return poids_total <= args_contraintes['capacite']

class GWO:
    def __init__(self, fonction_objective, contraintes, args_contraintes, nb_loups, max_iterations):
        self.fonction_objective = fonction_objective
        self.contraintes = contraintes
        self.args_contraintes = args_contraintes
        self.dim = len(args_contraintes['poids'])
        self.nb_loups = nb_loups
        self.max_iterations = max_iterations
        self.evolution_fitness = []
        self.best_fitness = -np.inf

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
            
            # Assurer que la meilleure fitness est croissante
            if fitness >= self.best_fitness:
                self.best_fitness = fitness
            else:
                # Rétablir la dernière meilleure position si la fitness n'a pas augmenté
                position[self.nb_loups - 1] = self.last_best_position

            self.evolution_fitness.append(self.best_fitness)

        return self.top_three_solutions(position)

    def mise_a_jour_position(self, position, a, objets_tries):
        positions_set = {tuple(p) for p in position}

        for i in range(self.nb_loups):
            if len(objets_tries) < 3:
                break

            fitness = [self.fonction_objective(position[j], self.args_contraintes['profits'], self.args_contraintes['repartition'], self.args_contraintes['nombre_de_variable']) for j in range(self.nb_loups)]
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
                    for j in objets_tries[::-1]:  # Enlever les objets ayant le plus bas profit/poids ratio
                        if new_position[j] == 1:
                            new_position[j] = 0
                            poids_total -= self.args_contraintes['poids'][j]
                        if poids_total <= self.args_contraintes['capacite']:
                            break

            positions_set.add(tuple(new_position))
            position[i] = new_position
            if fitness[i] >= self.best_fitness:
                self.last_best_position = position[i].copy()  # Mettre à jour la dernière meilleure position

        return position, a

    def meilleures_solutions(self, position):
        fitness = [self.fonction_objective(position[i], self.args_contraintes['profits'], self.args_contraintes['repartition'], self.args_contraintes['nombre_de_variable']) for i in range(self.nb_loups)]
        trier = np.argsort(fitness)[::-1]
        X_alpha = position[trier[0]].copy()
        alpha_fitness = fitness[trier[0]]
        X_beta = position[trier[1]].copy()
        beta_fitness = fitness[trier[1]]
        X_gamma = position[trier[2]].copy()
        gamma_fitness =         gamma_fitness = fitness[trier[2]]
        poids_total_alpha = np.sum(self.args_contraintes['poids'] * X_alpha)
        poids_total_beta = np.sum(self.args_contraintes['poids'] * X_beta)
        poids_total_gamma = np.sum(self.args_contraintes['poids'] * X_gamma)
        return X_alpha, alpha_fitness, poids_total_alpha, X_beta, beta_fitness, poids_total_beta, X_gamma, gamma_fitness, poids_total_gamma

    def top_three_solutions(self, position):
        fitness = [self.fonction_objective(position[i], self.args_contraintes['profits'], self.args_contraintes['repartition'], self.args_contraintes['nombre_de_variable']) for i in range(self.nb_loups)]
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
        return (X_alpha, alpha_fitness, poids_total_alpha), (X_beta, beta_fitness, poids_total_beta), (X_gamma, gamma_fitness, poids_total_gamma)

def afficher_positions(loup, repartition):
    start = 0
    for i, classe in enumerate(repartition):
        end = start + classe
        print(f"Classe {i + 1}: {loup[start:end]}")
        start = end

def afficher_nombre_variables_par_classe(position, repartition):
    start = 0
    for i, classe in enumerate(repartition):
        end = start + classe
        variables_classe = position[start:end]
        nombre_variables = np.sum(variables_classe)
        print(f"Classe {i + 1}: {nombre_variables} variables")
        start = end

def charger_donnees():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    chemin_fichier = filedialog.askopenfilename(title="Sélectionner le fichier de données")
    return chemin_fichier

# Charger les données depuis le fichier sélectionné
fichier = charger_donnees()
if fichier:
    nombre_de_variable, nombre_de_classe, capacite, repartition, profits, poids = lire_donnees(fichier)

    # Configuration de l'algorithme
    gwo = GWO(fonction_objective, contraintes, {
        'poids': poids,
        'profits': profits,
        'capacite': capacite,
        'repartition': repartition,
        'nombre_de_variable': nombre_de_variable
    }, nb_loups=30, max_iterations=15)
    top_solutions = gwo.initialisation()

    print("Meilleure solution:")
    afficher_positions(top_solutions[0][0], repartition)
    print("Fitness:", top_solutions[0][1])
    print("Poids total:", top_solutions[0][2])
    print()
    print("Deuxième meilleure solution:")
    afficher_positions(top_solutions[1][0], repartition)
    print("Fitness:", top_solutions[1][1])
    print("Poids total:", top_solutions[1][2])
    print()
    print("Troisième meilleure solution:")
    afficher_positions(top_solutions[2][0], repartition)
    print("Fitness:", top_solutions[2][1])
    print("Poids total:", top_solutions[2][2])

    print("\nNombre de variables dans chaque classe pour la meilleure solution:")
    afficher_nombre_variables_par_classe(top_solutions[0][0], repartition)

    # Afficher le graphique de l'évolution de la fitness
    plt.plot(gwo.evolution_fitness, label="Fitness")
    plt.title("Évolution de la fitness en fonction des itérations")
    plt.xlabel("Itération")
    plt.ylabel("Fitness")
    plt.xticks(np.arange(1, gwo.max_iterations + 1))  # Afficher toutes les itérations
    
    min_fitness = min(gwo.evolution_fitness)
    max_fitness = max(gwo.evolution_fitness)
    y_ticks = np.linspace(min_fitness, max_fitness, num=12, endpoint=False)  # Créer 10 graduations entre la valeur minimale et maximale de la fitness
    plt.yticks(y_ticks)  # Configurer l'axe y avec les valeurs de fitness calculées
    
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)  # Quadrillage horizontal
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Légende à l'extérieur du graphique
    plt.show()

