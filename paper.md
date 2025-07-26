---
title: "Approximate Clustering for Image Segmentation Using K-Means on Synthetic Data"
authors:
  - name: R. Charles Marchee
    orcid: https://orcid.org/0009-0007-2336-7162
    affiliation: Capella University
    email: rmarchee@capellauniversity.edu
version: 1.0.0
license: "Apache‑2.0"
repository: "https://github.com/r-charles-marchee/kmeans-synthetic-image-segmentation"

date: 2025-07-26
---


Summary
This software demonstrates the application of exact and approximate Principal Component Analysis (PCA) for dimensionality reduction on the scikit-learn Digits dataset. The objective is to reduce the feature dimensionality to two principal components using either exact PCA based on full singular value decomposition or approximate PCA using randomized singular value decomposition. Logistic regression models are trained on the full standardized dataset as well as on the PCA-reduced datasets. Classification accuracies and computational runtimes are compared to evaluate the trade-offs between predictive performance and efficiency. While logistic regression trained on the full dataset achieves higher classification accuracy, approximate PCA provides a computationally efficient alternative to exact PCA, offering a balance between speed and performance. The software also includes visualization of the approximate PCA projections colored by digit labels to support interpretation.

Statement of Need
Principal Component Analysis is a fundamental technique in machine learning for reducing dimensionality, simplifying datasets, and mitigating computational costs. Exact PCA, which relies on full singular value decomposition, can be computationally demanding for large datasets or high-dimensional data. Approximate PCA methods, such as randomized matrix factorization, offer reduced computational complexity at the potential expense of accuracy. Despite the theoretical development of approximate PCA, accessible examples illustrating its practical impact on classification tasks remain limited. This software addresses this gap by providing a reproducible example that applies both exact and approximate PCA on a canonical dataset. Logistic regression classifiers are trained on the resulting datasets, and both classification performance and computational time are compared. This example supports researchers and educators in investigating the balance between computational efficiency and model accuracy in dimensionality reduction.

Implementation
The software is implemented in Python 3, utilizing established scientific libraries including scikit-learn for data handling, preprocessing, PCA computation, and logistic regression modeling. Numerical computations are performed with NumPy, and Matplotlib is employed for visualization. The workflow begins with loading the scikit-learn Digits dataset, which consists of 1797 samples of 8×8 pixel images representing handwritten digits from 0 through 9. The dataset is partitioned into training and testing subsets, and feature values are standardized to zero mean and unit variance prior to PCA. Two PCA variants are applied: exact PCA using the full singular value decomposition solver and approximate PCA using randomized singular value decomposition. Both methods reduce the feature space to two principal components. Logistic regression models are trained separately on the full standardized dataset, the exact PCA-reduced data, and the approximate PCA-reduced data. Classification accuracy and computational time are measured and reported for each model. Visualization includes a scatter plot of the training data projected onto the first two principal components obtained via approximate PCA, with points colored according to their digit labels.

Usage
After installing the required dependencies, including scikit-learn, numpy, and matplotlib, users can execute the main Python script. The script automatically downloads the Digits dataset, performs data standardization, computes both exact and approximate PCA transformations, trains logistic regression models on the original and reduced datasets, and evaluates classification accuracy and PCA runtime. Output includes performance metrics printed to the console and a visualization of the approximate PCA projection of training samples. This minimal, complete example facilitates reproducibility and exploration of the trade-offs between dimensionality reduction computational efficiency and classification accuracy.

Impact
This software provides a concise demonstration of the differences between exact and approximate PCA methods applied to a standard classification task. By comparing classification accuracy with computational time, it enables users to assess when approximate PCA methods may serve as appropriate alternatives for computationally intensive dimensionality reduction. The software is intended to support machine learning practitioners, educators, and students in developing a deeper understanding of PCA techniques, randomized algorithms for matrix factorization, and their integration with classification workflows. The implementation can be extended to other datasets, dimensionality reduction techniques, or classifiers, providing a foundation for further research and instruction.

References
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Duchesnay, É., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830. Available at http://jmlr.org/papers/v12/pedregosa11a.html.

Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2), 217–288. https://doi.org/10.1137/090771806.
