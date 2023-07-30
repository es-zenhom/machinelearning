Welcome to my GitHub repository, where I host all my data analysis projects completed as part of my journey in acquiring advanced skills in data analysis. This repository is structured around several exercises that touch on crucial data analysis techniques such as Decision Trees, Random Forests, Machine Learning models like SVMs, Naive Bayes, and various visualization techniques. 

## Particle Track Identification with Graph Neural Networks
This project, collaborated on with Mathew Dittrich, involves the application of Graph Neural Networks (GNN) in the field of particle physics, specifically for the task of identifying true and false particle tracks in a tracker. The particle trajectories in a tracker were modelled using mini-doublets (MDs) and line segments (LSs). We used a PU 200 ttbar sample for initial training and the 96th event was used as a test case. 

Training involved three distinct Multi-Layer Perceptrons (MLPs) to compute and classify the LS. Post-training, we conducted inference testing and established a preliminary cut-off point, observing a reduction in fake edges and retention of true edges. Future work will revolve around optimizing the cut value, investigating MLP layer messages, data normalization, and adjusting the learning rate. This project illustrates the promise of deep learning techniques, specifically GNN, in the complex field of particle physics. Further optimization and development of these techniques could lead to significant advancements in this research area.

## Decision Trees and Random Forests Challenges
The notebook `DecisionTreesChallenges (2) (2).ipynb` consists of three challenges that explore the application of Decision Trees and Random Forests using Digits and Iris datasets. It demonstrates the workings of these methods, feature importance extraction, and visualization of decision trees.

## NumPy Exercises
A set of NumPy exercises demonstrating various operations on arrays, properties of matrices, and more, can be found in the notebook `NumPyExercises.ipynb`.

## Machine Learning: Clustering Challenges
The notebook `ClusteringChallenges.ipynb` contains exercises related to different clustering techniques - K-means, Gaussian Mixture Models (GMM), and Silhouette Analysis. It demonstrates how clustering can help identify common patterns in data, generate new data, and choose the optimal number of clusters.

## Linear Regression Challenges
In the notebook `LinearRegressionChallenges.ipynb`, I perform various tasks related to Linear Regression using the Boston Housing Prices Dataset, illustrating feature selection, interpretation of model coefficients, and detection of overfitting.

## Manifold Learning Challenges
The `ManifoldLearningChallenges (1) (2).ipynb` notebook provides a comprehensive demonstration of manifold learning methods applied to the Swiss roll dataset using methods such as Multidimensional Scaling (MDS), Locally Linear Embedding (LLE), and Isomap.

## Data Visualization: Matplotlib and Seaborn
`MatplotlibDatasetExercises (1).ipynb` contains exercises on data visualization using matplotlib and seaborn libraries, demonstrating how to create different types of plots including line, scatter, and histogram.

## Naive Bayes Classification
The exercise `NaiveBayesExercise.ipynb` focuses on the Naive Bayes algorithm for text and numerical data classification using Multinomial Naive Bayes and Gaussian Naive Bayes models.

## Feature Scaling
`FeatureScalingExercise.ipynb` is a notebook showcasing the concept of feature scaling, specifically min-max scaling, an essential preprocessing step in machine learning.

## Support Vector Machines (SVM) Challenges
In `SVMChallenges.ipynb`, various challenges related to the Support Vector Machines (SVM) algorithm are completed, such as face recognition, and classification of a moons dataset using Linear SVC, Polynomial Kernel, and Radial Basis Function (RBF) Kernel.

Each notebook provides an overview of the concepts, followed by a step-by-step guide on how to apply these techniques using real-world datasets. Feel free to explore the notebooks for a detailed walkthrough.

## Installation
To run these notebooks, you need to have Python and the necessary Python packages installed. You can install these packages using `pip`:

```
pip install numpy pandas sklearn matplotlib seaborn
```

Or if you prefer, you can use `Anaconda`, a pre-packaged Python distribution that contains all the necessary libraries for data analysis and scientific computing.

Thank you for visiting my repository. If you have any questions or would like to discuss these projects further, please feel free to contact me.
