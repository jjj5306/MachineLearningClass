import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = "assignment2/tiger.jpg"

# K-means parameters setting
k = [2, 5, 10]
n = 10 # max iteration

# Process tiger image
img = Image.open(img_path)
img_array = np.array(img)
height, width, channel = img_array.shape

# result figure setting
plt.figure(figsize=(15, 5))
plt.suptitle("Tiger Image Segmentation with K-means", fontsize=16)

# Process with different k
for i, k in enumerate(k):
    # Reshape image to (n_pixels, n_features)
    img_data = img_array.reshape(height*width, channel)
    
    # Randomly initialize centroids
    samples_len = img_data.shape[0]
    randomly_picked = np.random.choice(samples_len, k, replace=False) # 중복 없이 k개의 인덱스를 무작위로 선택함
    centroids = img_data[randomly_picked]
    
    # Iterate for clustering
    for j in range(n):
        print(f"K-means iteration {j+1}/{n}")
        
        # Calculate distance each point to centroids
        distances = np.zeros((img_data.shape[0], k))
        for c in range(k):
            distances[:, c] = np.sqrt(np.sum((img_data - centroids[c])**2, axis=1))
        # 각 픽셀별로 가장 가까운 centroid를 찾아서 label로 지정함.
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([img_data[labels == c].mean(axis=0) if np.sum(labels == c) > 0 
                                 else centroids[c] for c in range(k)])
        
        # case condition
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    # Mapping each pixel to its cluster center
    segmented_img = centroids[labels].reshape(height, width, channel).astype(np.uint8)
    
    # Plot result
    plt.subplot(1, 3, i+1)
    plt.imshow(segmented_img)
    plt.title(f"K = {k}")

plt.tight_layout()
plt.savefig("assignment2/tiger_segmentation.png")