import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Reconstruction des données de ton Tableau 1
# J'ai recopié les valeurs exactes de ton article
data = {
    'Send to Node-10': [
        -1.7737, -7.8649, -1.265, -1.6023, 0, -2.3739,
        -1.9825, -7.8622, -2.5827, -1.8968, -0.6, -2.5526
    ],
    'Send to Node-3': [
        -1.09, -7.8416, -2.6258, -0.9, 0, -1.755,
        -1.5413, -7.8390, -3.7555, -3.1675, -2.0557, -2.0258
    ]
}

# Les index correspondent aux "Target Nodes" (Lignes)
targets = [f"Node-{i}" for i in range(12)]

# Création du DataFrame
df = pd.DataFrame(data, index=targets)

# 2. Configuration du style
sns.set_theme(style="white")
plt.figure(figsize=(8, 10)) # Format portrait adapté à une longue liste

# 3. Création de la Heatmap
# 'RdYlGn' est une colormap intuitive : Rouge (négatif/coûteux) -> Vert (proche de 0/bon)
heatmap = sns.heatmap(df,
            annot=True,       # Affiche les chiffres
            fmt=".2f",        # Arrondi à 2 décimales pour la lisibilité
            cmap="RdYlGn",    # Rouge à Vert
            linewidths=.5,    # Petites lignes blanches entre les cases
            cbar_kws={'label': 'Q-Value (estimated cumulative reward, with discount)'}
           )

# 4. Ajout de titres et labels
plt.title('Q-Matrix for the Agent "Node-4"\n(Estimate of the discounted cost for each target node)', fontsize=14, pad=20)
plt.ylabel('Target Node', fontsize=12)
plt.xlabel('Action (chosen neighbor)', fontsize=12)

# Ajustement pour que les labels du bas soient bien lisibles
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# 5. Sauvegarde ou Affichage
plt.tight_layout()
plt.show()
# Pour sauvegarder l'image en haute qualité pour ton article :
# plt.savefig('q_matrix_heatmap.png', dpi=300)