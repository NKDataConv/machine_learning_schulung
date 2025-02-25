import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=1000, n_features=20,
                           n_classes=2, random_state=42,
                           n_informative=10, n_redundant=5)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_score = clf.predict_proba(X_test) # [:, 1]
y_score = y_score[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

fig = go.Figure()

# ROC Curve hinzufügen
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                         name=f'ROC Curve (AUC = {roc_auc:.2f})',
                         line=dict(color='blue', width=2)))

# Diagonale Linie (Random Classifier)
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                         name='Random Guess',
                         line=dict(color='gray', dash='dash')))

# Layout anpassen
fig.update_layout(title='ROC Curve',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  showlegend=True,
                  width=700, height=500)

# Plot anzeigen
fig.show()