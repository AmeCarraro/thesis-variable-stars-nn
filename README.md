Neural Network Approaches for Classifying Variable Star Light Curves

📍 University of Padova  
🎓 Author: Amedeo Carraro | 👨‍🏫 Supervisor: Prof. Loris Nanni | 📅 Defense: 24/03/2025


✨ Overview
Welcome to the GitHub repository accompanying my master's thesis: a deep dive into the stars through machine learning. 
This project focuses on the classification of variable stars—like Cepheids, BY Draconis, Delta Scuti, and eclipsing binaries—by analyzing their light curves, using two main deep learning approaches:
A Long Short-Term Memory (LSTM) model to capture temporal dependencies in reconstructed light curves.
A Convolutional Neural Network (CNN) trained on image-like transformations of the light curves (including DMDT representations).

🗂 Dataset
The models are trained and evaluated on real astronomical data from the Zwicky Transient Facility (ZTF).
Training: 5-fold cross-validation, 5-class and 10-class configurations.

All models were evaluated using:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix

💡Results show strong classification performance from both models, with the CNN excelling in capturing class-specific visual features and the LSTM showing robustness to sequence-based data.
