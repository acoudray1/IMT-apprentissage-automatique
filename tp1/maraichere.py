# Importer le module tree de la librairie sklearn
import pydot as pydot
from sklearn import tree
from io import StringIO

RUG = 0
LIS = 1
POM = "pomme"
ORA = "orange"
feature_names = ["poids", "texture"]
target_names = [POM, ORA]

# Declarer le tableau des caracteristiques
carac = [
    [140, LIS],
    [130, LIS],
    [150, RUG],
    [170, RUG]
]

# Declarer le tableau des etiquettes
etiq = [POM, POM, ORA, ORA]

# Appeler la methode tree.DecisionTreeClassifier() et stocker le resultat dans clf

# On a serious note, random_state simply sets a seed to the random generator, so that
# your train-test splits are always deterministic. If you don't set a seed, it is different each time.
clf = tree.DecisionTreeClassifier(random_state=23)

# Construire l’arbre de decision a partir des donnees d’entrainement
clf.fit(carac, etiq)

# Prediction sur un couple d'entree
res = clf.predict([[143, LIS]])

print(res)

# Recuperation du graph (pdf)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=feature_names,
                     class_names=target_names,
                     filled=True, rounded=True, impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("fruits.pdf")
