import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(image):
    features = base_model.predict(image)
    return features.flatten()

# Nature-inspired optimization algorithms

# PSO
def particle_swarm_optimization(features, target, n_particles=30, max_iter=100):
    n_features = features.shape[1]
    
    # Initialize particles
    particles = np.random.rand(n_particles, n_features)
    velocities = np.zeros((n_particles, n_features))
    personal_best = particles.copy()
    global_best = particles[np.random.choice(n_particles)]
    
    for _ in range(max_iter):
        for i in range(n_particles):
            # Update velocity
            r1, r2 = np.random.rand(2)
            velocities[i] = (0.5 * velocities[i] + 
                             1.5 * r1 * (personal_best[i] - particles[i]) + 
                             1.5 * r2 * (global_best - particles[i]))
            
            # Update position
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)
            
            # Update personal best
            if np.sum(particles[i]) < np.sum(personal_best[i]):
                personal_best[i] = particles[i]
            
            # Update global best
            if np.sum(particles[i]) < np.sum(global_best):
                global_best = particles[i]
    
    # Apply the best mask to features
    return features * global_best

# ABC
import numpy as np

def artificial_bee_colony(features, target, n_bees=30, max_iter=100, epsilon=1e-10):
    n_features = features.shape[1]
    
    # Initialize food sources
    food_sources = np.random.rand(n_bees, n_features)
    fitness = np.zeros(n_bees)
    
    for iter_count in range(max_iter):
        # Employed bees phase
        for i in range(n_bees):
            new_source = food_sources[i] + np.random.uniform(-1, 1, n_features)
            new_source = np.clip(new_source, 0, 1)
            if np.sum(new_source) < np.sum(food_sources[i]):
                food_sources[i] = new_source
        
        # Update fitness
        fitness = np.sum(food_sources, axis=1)
        
        # Print the fitness values for debugging
        print(f"Iteration {iter_count}: Fitness values = {fitness}")
        
        # Ensure fitness contains valid values
        if np.isnan(fitness).any() or np.isinf(fitness).any():
            print("Invalid fitness values detected")
            fitness = np.nan_to_num(fitness, nan=epsilon, posinf=epsilon, neginf=epsilon)
        
        # Ensure no division by zero in probabilities
        fitness_sum = np.sum(fitness)
        if fitness_sum == 0 or np.isnan(fitness_sum):
            print("Fitness sum is zero or NaN, using epsilon")
            fitness_sum = epsilon  # Avoid division by zero
        
        probabilities = fitness / fitness_sum
        
        # Print the probabilities for debugging
        print(f"Iteration {iter_count}: Probabilities = {probabilities}")
        
        # Onlooker bees phase
        for _ in range(n_bees):
            if np.isnan(probabilities).any():
                print("Invalid probabilities encountered, resetting to uniform distribution")
                probabilities = np.ones(n_bees) / n_bees  # Uniform distribution if invalid
            
            i = np.random.choice(n_bees, p=probabilities)
            new_source = food_sources[i] + np.random.uniform(-1, 1, n_features)
            new_source = np.clip(new_source, 0, 1)
            if np.sum(new_source) < np.sum(food_sources[i]):
                food_sources[i] = new_source
        
        # Scout bees phase
        worst = np.argmax(fitness)
        food_sources[worst] = np.random.rand(n_features)
        
        # Update fitness after scout phase
        fitness = np.sum(food_sources, axis=1)
    
    best_source = food_sources[np.argmin(fitness)]
    return features * best_source


# BCS
def binary_cuckoo_search(features, target, n_nests=25, max_iter=100):
    n_features = features.shape[1]
    
    # Initialize nests
    nests = np.random.rand(n_nests, n_features) > 0.5
    fitness = np.sum(nests, axis=1)
    
    for _ in range(max_iter):
        # Generate new solutions
        new_nests = nests.copy()
        for i in range(n_nests):
            step = np.random.randn(n_features)
            new_nests[i] = nests[i] + step > 0.5
        
        # Update if better
        new_fitness = np.sum(new_nests, axis=1)
        improved = new_fitness < fitness
        nests[improved] = new_nests[improved]
        fitness[improved] = new_fitness[improved]
        
        # Abandon worst nests
        worst = np.argsort(fitness)[-n_nests//5:]
        nests[worst] = np.random.rand(len(worst), n_features) > 0.5
        fitness[worst] = np.sum(nests[worst], axis=1)
    
    best_nest = nests[np.argmin(fitness)]
    return features * best_nest

# Directory for images
image_dir = r"C:\Users\LENOVO\Desktop\BTP\model\glaucoma_detection\images"
data = [
    os.path.join(image_dir, "image1.jpg"),
    os.path.join(image_dir, "image2.jpg"),
    os.path.join(image_dir, "image3.jpg"),
    os.path.join(image_dir, "image4.jpg")
]
labels = [0, 1, 0, 1]

# Feature extraction
extracted_data = []
for image_path in data:
    try:
        image = preprocess_image(image_path)
        features = extract_features(image)
        extracted_data.append(features)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if len(extracted_data) != len(labels):
    print(f"Warning: Number of processed images ({len(extracted_data)}) doesn't match number of labels ({len(labels)})")
    labels = labels[:len(extracted_data)]

# Convert extracted_data to a numpy array
extracted_data = np.array(extracted_data)

# Apply optimization algorithms
X_train_pso = particle_swarm_optimization(extracted_data, labels)
X_train_abc = artificial_bee_colony(X_train_pso, labels)
X_train_bcs = binary_cuckoo_search(X_train_abc, labels)

# Use all data for training
X_train = X_train_bcs
y_train = np.array(labels)

# Debugging output
print("Unique classes in y_train:", np.unique(y_train))
print("Class counts in y_train:", np.bincount(y_train))

# Initialize classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, random_state=42)

# Ensemble model
ensemble_model = VotingClassifier(estimators=[('rf', rf), ('svm', svm)], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)
print("Ensemble model trained")

# Testing function
def test_new_image(image_path):
    new_image = preprocess_image(image_path)
    new_image_features = extract_features(new_image)
    new_image_features = new_image_features.reshape(1, -1)
    new_image_features_pso = particle_swarm_optimization(new_image_features, [0])
    new_image_features_abc = artificial_bee_colony(new_image_features_pso, [0])
    new_image_features_bcs = binary_cuckoo_search(new_image_features_abc, [0])
    prediction = ensemble_model.predict(new_image_features_bcs)
    return 'Glaucoma' if prediction[0] == 1 else 'Healthy'

# classification report
print(classification_report(labels, ensemble_model.predict(X_train)))

# accuracy score
print(accuracy_score(labels, ensemble_model.predict(X_train)))

# Test the model
print("\nTesting the model:")
for i, image_path in enumerate(data):
    result = test_new_image(image_path)
    print(f"Sample {i+1}: Actual label = {labels[i]}, Prediction = {result}")

print("\nNote: This model is trained on very limited data and may not be reliable for real-world use.")
