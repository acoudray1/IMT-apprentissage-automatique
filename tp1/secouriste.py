import pandas as pd
from sklearn import tree

# Chargement des donnees
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("assets/titanic.csv", sep=';')

# Extraction des colonnes pertinentes et convertion des chaines de caracteres en entier
df = df[['pclass', 'survived', 'sex', 'age']]
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Suppression des donnees incompletes
df = df.dropna()

feat = df[['pclass', 'sex', 'age']]
target = df[['survived']]

# Construire arbre de décision ...
clf = tree.DecisionTreeClassifier(random_state=1093)
X_train, X_test, y_train, y_test = train_test_split(feat, target)
clf.fit(X_train, y_train)

# Calculer la qualite de prediction sur l'arbre de decision sur les donnes de tests
predictQuality = clf.score(X_test, y_test)
# En limitant le nombre de feuilles a 3 on reduit la qualite de prediction
print("qualite d'entrainement : " + str(predictQuality))
# Observer les differences en fonction du random_state
scores = cross_val_score(clf, feat, target, cv=5)
print("accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_predict = clf.predict(X_test)
# Afficher la matrice de confusion ...
cfm = confusion_matrix(y_test, y_predict)
pd.DataFrame(
    cfm,
    columns=['Mort prevue', 'Survie prevue'],
    index=['Mort', 'Survivant']
)

print(cfm)
