# ==================================================
# 1. Import der notwendigen Bibliotheken
# ==================================================
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ================================
# 2. Daten laden
# ================================
# Der Iris-Datensatz enthält vier Merkmale pro Blüte (Sepal Length, Sepal Width,
# Petal Length, Petal Width) und drei Klassen (Setosa, Versicolor, Virginica).
iris = load_iris()
X = iris.data           # Merkmalsmatrix (4 Dimensionen)
y_true = iris.target    # Die wahren Klassenlabels, NUR zur Auswertung/Illustration


# ================================
# 3. Daten skalieren (optional)
# ================================
# Clustering profitiert häufig von Daten, die im ähnlichen Wertebereich liegen.
# Wir skalieren daher alle 4 Features. (Die Visualisierung erfolgt jedoch später
# auf den Originalwerten der ersten beiden Features.)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 4D-Daten für das Clustering


# ================================
# 4. KMeans Clustering
# ================================
# Wir initialisieren KMeans mit:
#  - n_clusters = 3 (da wir wissen, dass Iris theoretisch 3 Arten hat)
#  - n_init = 10 (KMeans wird 10-mal neu gestartet; das beste Ergebnis wird gewählt)
#  - random_state = 42 (zur Reproduzierbarkeit)
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X_scaled)

# Die Cluster-Zugehörigkeiten für jeden Datenpunkt
cluster_labels = kmeans.labels_


# ================================
# 5. Auswertung: Silhouette Score
# ================================
# Der Silhouette-Score misst, wie gut die Punkte in ihren Clustern liegen.
# Werte reichen von -1 bis 1, wobei höhere Werte eine bessere Clustertrennung bedeuten.
sil_score = silhouette_score(X_scaled, cluster_labels)
print("Silhouette Score:", sil_score)


# ================================
# 6. Visualisierung mit Plotly
# ================================
# Da das Iris-Datenset 4-dimensionale Merkmale hat,
# wählen wir zwei Features (z.B. die ersten beiden: 'Sepal Length' und 'Sepal Width'),
# um einen 2D-Scatterplot zu erstellen.
# - Farbe = KMeans-Cluster
# - Symbol oder Farbe = Wahre Klasse in einem separaten Plot

# Wir definieren hier zur Übersicht Feature-Namen:
feature_names = iris.feature_names  # ["sepal length (cm)", "sepal width (cm)", ...]
feature_x = 0  # Index für 'Sepal Length'
feature_y = 1  # Index für 'Sepal Width'

# Plot 1: KMeans-Cluster auf Original-Features (Index 0 und 1)
fig1 = px.scatter(
    x=X[:, feature_x],
    y=X[:, feature_y],
    color=cluster_labels.astype(str),  # Cluster-Labels als String für kategorische Farbe
    title="KMeans-Cluster auf Iris-Daten (Sepal-Length vs. Sepal-Width)",
    labels={
        "x": feature_names[feature_x],
        "y": feature_names[feature_y],
        "color": "KMeans Cluster"
    }
)
fig1.update_layout(template='plotly_white')
fig1.show()

# Plot 2: Wahre Klassenlabels auf denselben Features
fig2 = px.scatter(
    x=X[:, feature_x],
    y=X[:, feature_y],
    color=y_true.astype(str),  # Wahre Klassenlabels als String für kategorische Farbe
    title="Wahre Klassen (Sepal-Length vs. Sepal-Width)",
    labels={
        "x": feature_names[feature_x],
        "y": feature_names[feature_y],
        "color": "Wahre Klasse"
    }
)
fig2.update_layout(template='plotly_white')
fig2.show()


# ==================================================
# ANMERKUNGEN:
# --------------------------------------------------
# 1. Für das eigentliche Clustering werden alle 4 Features
#    (nach Skalierung) herangezogen, da dies bessere Ergebnisse
#    liefern kann, als nur 2 Features zu nutzen.
#
# 2. Bei der Visualisierung betrachten wir zur Veranschaulichung
#    nur 2 Features (Sepal Length, Sepal Width) in ihrer Originalskala.
#    Daher kann es sein, dass die Grenzen zwischen den Clustern
#    in der 2D-Darstellung nicht optimal erscheinen, obwohl das
#    Clustering auf allen 4 Features erfolgt.
#
# 3. Der Silhouette Score liefert uns eine Metrik für die Güte
#    des Clusters auf Basis aller 4 Features.
#
# 4. KMeans ist nicht immer die beste Wahl, speziell wenn die
#    Cluster keine kugelförmige Struktur haben oder wenn Ausreißer
#    stark vorhanden sind. Andere Verfahren (z.B. DBSCAN,
#    hierarchische Clusteranalyse) können in solchen Fällen
#    überlegen sein.
#
# 5. Im Code wurden einige fortgeschrittene Konzepte genutzt,
#    die nicht weiter erläutert wurden (z.B. Silhouette Score,
#    StandardScaler, bestimmte KMeans-Parameter).
# ==================================================