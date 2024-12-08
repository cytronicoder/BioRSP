from setuptools import setup, find_packages

# Read the contents of the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="biorsp",
    version="0.1.0",
    author="Zeyu Yao",
    author_email="cytronicoder@gmail.com",
    description="A computational tool for analyzing embedding patterns in scRNA-seq data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cytronicoder/biorsp",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.1.0",
        "matplotlib>=3.9.0",
        "scipy>=1.14.0",
        "pandas>=2.2.0",
        "seaborn>=0.13.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.6",
    keywords="scRNA-seq, bioinformatics, data analysis, single-cell, RNA sequencing",
    license="Apache License 2.0",
    project_urls={
        "Documentation": "https://github.com/yourusername/bioRSP#readme",
        "Source": "https://github.com/yourusername/bioRSP",
        "Bug Tracker": "https://github.com/yourusername/bioRSP/issues",
    },
)
