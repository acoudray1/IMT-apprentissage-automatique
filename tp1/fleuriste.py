from io import StringIO
import pydot as pydot
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score

# Chargement du jeu de donnees
iris = load_iris()
X = iris.data
y = iris.target

# Separer les donnees en exemples d'entrainements et de tests
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Construire et entrainer un arbre de decision sur les donnees d'entrainement
clf = tree.DecisionTreeClassifier(random_state=3, max_leaf_nodes=3)
clf.fit(X_train, y_train)

# Calculer la qualite de prediction sur l'arbre de decision sur les donnes de tests
predictQuality = clf.score(X_test, y_test)
# En limitant le nombre de feuilles a 3 on reduit la qualite de prediction
print("qualite d'entrainement : " + str(predictQuality))

# Observer les differences en fonction du random_state
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print("accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Recuperation du graph (pdf)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     filled=True, rounded=True, impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf")