# Customer Segmentation using K-means clustering
This project uses K-means clustering to group customers based on their spending behaviors. The Mall Customer Segmentation dataset was used to train the model, which was implemented in TensorFlow and Keras.

## Dataset
The dataset contains information on customer demographics and spending habits, such as age, gender, annual income, and spending score.

## Approach
K-means clustering is an unsupervised machine learning algorithm that partitions a given dataset into K clusters. The algorithm aims to minimize the distance between the data points and the centroid of the cluster they belong to.

In this project, the K-means algorithm was used to cluster customers based on their annual income and spending score. The optimal number of clusters was determined using the elbow method.

## Results
The performance of the clustering was evaluated using the silhouette score and within-cluster sum of squares (WSS). The best number of clusters was selected using the elbow method. The results showed that the customers could be segmented into distinct groups based on their spending behavior. The results were visualized using a scatter plot, which clearly showed the segmentation of customers into different groups.

## Usage
To run the project, clone the repository and run the customer_segmentation.ipynb notebook. Make sure to install the necessary dependencies using the requirements.txt file.

## Future Work
This project can be extended by incorporating additional features, such as customer reviews or social media activity, to improve the clustering accuracy. Another possible extension is to evaluate the performance of different clustering algorithms and compare their results.

## References
Mall Customer Segmentation Dataset: https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
Scikit-learn documentation: https://scikit-learn.org/stable/index.html
K-means clustering: https://en.wikipedia.org/wiki/K-means_clustering
