from setuptools import find_packages, setup

setup(
    name="actuator_modeling",
    version="0.1.0",
    description="GRU model for actuator torque prediction",
    author="NTNU Robotics",
    author_email="your.email@ntnu.no",
    python_requires=">=3.8.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "pytorch-lightning>=1.5.0",
        "torchmetrics>=0.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.7.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "export": ["onnx>=1.10.0", "onnxruntime>=1.10.0"],
        "dev": [
            "black",
            "isort",
            "flake8",
            "pytest",
            "pytest-cov",
            "mypy",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 