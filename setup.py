from setuptools import setup, find_packages

setup(
    name="nonergodic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "torch>=2.0",
        "transformer-lens>=2.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
    ],
)
