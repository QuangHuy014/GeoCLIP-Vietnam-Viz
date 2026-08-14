from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="geoclip-vietnam",
    version="1.0.0",
    author="GeoCLIP Vietnam Project",
    author_email="contact@geoclip-vietnam.org",
    description="Python library for image geo-localization in Vietnam and Globally using GeoCLIP (Vision-Location Matching)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QuangHuy014/GeoCLIP-Vietnam-Viz",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": [
            "weights/*.pth",
            "data/*.csv",
            "data/*.geojson",
            "data/sample_images/*"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pillow>=9.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "folium>=0.14.0",
        "streamlit>=1.25.0"
    ],
)
