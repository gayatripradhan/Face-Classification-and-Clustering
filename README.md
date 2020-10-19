# Face-Classification-and-Clustering


  
## Dependencies:
- numpy
- imutils
- sklearn
- argparse
- pickle
- openCV (cv2)
- os
- glob

# Classify_Known.py
encode_faces.py script will classify the faces to known faces.

**What it does**
- It will load the embedding vectors from "Data/train" and "Data/test" dicertory as train and test sets
- Then as normal practice it will normalize the embedding vectos of faces.
- The LabelEncoder class in scikit-learn will convert the labels into integers.
- Create a KNN object and then fit the model on the embeddings

**Usage - To run**
>python Classify_Known.py

# Classify_Known_Unknown.py
Classify_Known_Unknown.py script will classify the faces into known and unknown faces.

**Arguments:**
- -e --embeddings : The path to the embedding vectors (Data/train).
- -u --face_to_compare : The path to the embedding vectors (Data/Data for Known-Unknown test)
- -t --tolerance : Tolerance for the face distance measure

**What it does**
- It will load the embedding vectors from "Data/train" as know list of faces.
- For unknown list of faces it will load from "Data/Data for Known-Unknown test" directory.
- Calculates the face distance between the known face list & unknown face list and classifys it to "Known" and "Unknown".
- We employ the build_montages function of imutils to generate a single image montage containing a 5×5 grid of Known faces and unknown faces.

**Usage - To run**
>python Classify_Known_Unknown.py --embeddings Data/train --face_to_compare   Data/Data for Known-Unknown test --tolerance 0.8

Sample result:

![KNOWN](https://github.com/gayatripradhan/Face-Classification-and-Clustering/blob/main/results/Known.PNG)

![UNKNOWN](https://github.com/gayatripradhan/Face-Classification-and-Clustering/blob/main/results/Unknown.PNG)

# Clustering.py
We have embedding vectors of all faces in our dataset as 512-d vectors, the next step is to cluster them into groups.

For this task we need a clustering algorithm, many clustering algorithms such as k-means and Hierarchical 
Agglomerative Clustering, require us to specify the number of clusters.
Therefore, we need to use a density-based or graph-based clustering algorithm
*Density-based spatial clustering of applications with noise (DBSCAN)*

**Arguments:**
- -i --encodings : The path to the embeddings
- -d --jobs : DBSCAN is multithreaded and a parameter can be passed to the constructor containing the number of parallel jobs to run. 
              A value of -1 will use all CPUs available (default).

**What it does**
- create a DBSCAN object and then fit the model on the embeddings
- loop to over all the cluster ids.
- We employ the build_montages function of imutils to generate a single image montage containing a 5×5 grid of faces

**To run**
>python Clustering.py --encodings Data\train 

Sample Culters:

![Clusters](https://github.com/gayatripradhan/Face-Classification-and-Clustering/blob/main/results/Clusters.PNG)
