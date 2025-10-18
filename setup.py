#!/usr/bin/env python3
"""
Setup script for RWKV-Raven-Cog: OpenCog transformation for RWKV models
"""

from setuptools import setup, find_packages

setup(
    name="rwkv-raven-cog",
    version="0.1.0",
    description="OpenCog transformation for RWKV-4-Raven language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="OpenCog Community",
    author_email="opencog@example.com",
    url="https://github.com/cogpy/rwkv-raven-cog",
    packages=find_packages(),
    py_modules=["opencog_transform"],
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.21.0", 
        "huggingface_hub>=0.20.0",
        "transformers>=4.21.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "fsspec>=2023.5.0",
        "filelock>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "rwkv-opencog-transform=opencog_transform:main",
        ],
    },
)