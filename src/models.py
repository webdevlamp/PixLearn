from sklearn.svm import SVC

def create_svm_model():
    # SVC with RBF kernel is excellent for image classification
    return SVC(kernel='rbf', gamma='scale', C=10, probability=True)
