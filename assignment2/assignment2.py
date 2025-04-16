import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

tiger_path = "assignment2/tiger.jpg"

# K-means parameters setting
k_values = [2, 5, 10]
max_iterations = 10

# Process tiger image
img = Image.open(tiger_path)
img_array = np.array(img)
h, w, c = img_array.shape

# result figure setting
plt.figure(figsize=(15, 5))
plt.suptitle("Tiger Image Segmentation with K-means", fontsize=16)

# Process with different k values
for i, k in enumerate(k_values):
    print(f"Starting k-means with k={k}")
    
    # Reshape image to (n_pixels, n_features)
    img_data = img_array.reshape(h*w, c)
    
    # Randomly initialize centroids
    n_samples = img_data.shape[0]
    centroids_indices = np.random.choice(n_samples, k, replace=False)
    centroids = img_data[centroids_indices]
    
    # Iterate for clustering
    for iteration in range(max_iterations):
        print(f"K-means iteration {iteration+1}/{max_iterations}")
        
        # Calculate distance each point to centroids
        distances = np.sqrt(((img_data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([img_data[labels == j].mean(axis=0) if np.sum(labels == j) > 0 
                                 else centroids[j] for j in range(k)])
        
        # case condition
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    # Mapping each pixel to its cluster center
    segmented_img = centroids[labels].reshape(h, w, c).astype(np.uint8)
    
    # Plot result
    plt.subplot(1, 3, i+1)
    plt.imshow(segmented_img)
    plt.title(f"K = {k}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("assignment2/tiger_segmentation.png")