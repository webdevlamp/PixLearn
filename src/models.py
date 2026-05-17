from sklearn.svm import SVC

def create_svm_model():
    # Increased C (complexity) to better distinguish similar digits like 3 and 7
    return SVC(kernel='rbf', gamma='scale', C=100, probability=True)
