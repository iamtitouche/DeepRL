import matplotlib.pyplot as plt
import numpy as np

def draw_frozen_lake_policy(policy, lake_shape=(4, 4)):
    # Définir les directions des flèches (Right, Left, Up, Down)
    directions = {
        0: (0, 0.3),  # Right
        1: (0, -0.3), # Left
        2: (-0.3, 0), # Up
        3: (0.3, 0)   # Down
    }
    
    fig, ax = plt.subplots(figsize=(lake_shape[1], lake_shape[0]))
    
    # Configurer la grille
    ax.set_xticks(np.arange(0, lake_shape[1], 1))
    ax.set_yticks(np.arange(0, lake_shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    
    # Dessiner les flèches dans chaque case
    for i in range(lake_shape[0]):
        for j in range(lake_shape[1]):
            action = policy[i, j]
            dy, dx = directions[action]
            ax.arrow(j + 0.5, lake_shape[0] - i - 0.5, dx, dy, head_width=0.2, head_length=0.2, fc='k', ec='k')

    # Ajuster les limites pour centrer les flèches
    ax.set_xlim(0, lake_shape[1])
    ax.set_ylim(0, lake_shape[0])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    plt.show()

# Exemple de politique pour une grille 4x4 (0: droite, 1: gauche, 2: haut, 3: bas)
example_policy = np.array([
    [0, 0, 0, 0],
    [2, 2, 2, 2],
    [3, 1, 0, 0],
    [3, 3, 1, 0]
])

draw_frozen_lake_policy(example_policy)
