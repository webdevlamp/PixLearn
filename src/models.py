from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def create_sklearn_classifier(model_type='random_forest'):
    if model_type == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'svm':
        return SVC(kernel='rbf', probability=True)
    else:
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
