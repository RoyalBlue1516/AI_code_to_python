import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def generate_random_data(N, d, R=10, alpha=0.1):
    # Step 1: Random assignment of true parameters w, b
    w_true = np.random.uniform(-R, R, size=(d,))
    b_true = np.random.uniform(-R, R)
    
    # Step 2: Generate dataset
    X = np.random.uniform(-R, R, size=(N, d))
    gaussian = np.random.normal(0, alpha * R, size=(N,))
    y = np.dot(X, w_true) + b_true + gaussian
    
    return X, y

def split_dataset(X, y):
    # Step 3: Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.15, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42)
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def save_dataset(X_train, X_dev, X_test, y_train, y_dev, y_test, filename):
    # Step 4: Save dataset
    data = {
        'x_train': X_train,
        'x_dev': X_dev,
        'x_test': X_test,
        'y_train': y_train,
        'y_dev': y_dev,
        'y_test': y_test
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# Generate datasets with different sizes
N_values = [1000]

for N in N_values:
    X, y= generate_random_data(N, d=10)  # Assuming d=10 for example
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_dataset(X, y)
    filename = f"myrandomdataset.pkl"
    save_dataset(X_train, X_dev, X_test, y_train, y_dev, y_test, filename)