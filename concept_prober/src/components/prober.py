from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class Prober:

    def __init__(self):
        self.lr: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None

    def train(self, X, y, concept_to_class):
        y = [concept_to_class(k) for k in y]
        self.scaler = StandardScaler()
        X_transformed = self.scaler.fit_transform(X)

        self.lr = LogisticRegression(random_state=0, n_jobs=-1, multi_class='multinomial')

        self.lr.fit(X_transformed, y)

    def predict_class_for_each_instance(self, X):
        X_transformed = self.scaler.transform(X) # type: ignore
        return self.lr.predict(X_transformed) # type: ignore

    def predict_class_for_each_instance_proba(self, X):
        X_transformed = self.scaler.transform(X) # type: ignore
        return self.lr.predict_proba(X_transformed) # type: ignore





