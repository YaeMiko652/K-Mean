#from google.colab import drive
#drive.mount('/content/drive')

from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
new_image = imread("Kitty.jpg")

plt.figure(figsize=(20,20))
plt.imshow(new_image)
new_image = new_image/255.0
new_image.shape
X = new_image.reshape(-1,3)
X.shape
k_clusters = list(range(1,6))
sse = []

for k in k_clusters:
  km = KMeans(n_clusters=k)
  km.fit(X)
  sse.append(km.inertia_)

plt.figure(figsize=(10,10))
plt.plot(k_clusters, sse, '-o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Suqared Error')
plt.show()

km_new = KMeans(n_clusters=4)
km_new.fit(X)

img_seg = km_new.cluster_centers_
print(img_seg)

img_seg = img_seg[km_new.labels_]
img_seg = img_seg.reshape(new_image.shape)

plt.figure(figsize=(20,20))
plt.imshow(img_seg)