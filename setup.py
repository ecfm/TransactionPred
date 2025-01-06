from setuptools import setup, find_packages

setup(
    name="transaction_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "scikit-learn",
        "optuna",
        "mlflow",
        "hydra-core",
        "omegaconf",
        "tqdm",
        "gitpython",
    ],
)