from setuptools import find_packages, setup

setup(
    name="npyLinear",
    version="0.1.0",
    description="A simple Numpy based Linear Neural Network for classification tasks.",
    package_dir={"": "npyLinear"},
    packages=find_packages(where="npyLinear"),
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["matplotlib", "numpy", "pandas", "tqdm"],
    python_requires=">=3.9",
)
