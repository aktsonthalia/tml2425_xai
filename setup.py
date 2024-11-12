from setuptools import setup, find_packages

setup(
    name="tml2425_xai",  
    version="0.1",
    description="Utility functions for explainable AI, for the Trustworthy Machine Learning course at the University of Tübingen.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aktsonthalia/tml2425_xai",
    author="Ankit Sonthalia",
    author_email="ankit.sonthalia@uni-tuebingen.de",
    packages=find_packages(),
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
