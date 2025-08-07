from setuptools import setup, find_packages

setup(
    name="kmeans-synthetic-image-segmentation",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "scikit-image"
    ],
    author="R. Charles Marchee",
    author_email="rmarchee@capellauniversity.edu",
    description="Synthetic image segmentation using K-Means clustering",
    url="https://github.com/yourusername/kmeans-synthetic-image-segmentation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
