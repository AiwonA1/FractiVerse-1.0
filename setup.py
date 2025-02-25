from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fractiverse",
    version="1.0.0",
    author="FractiVerse Team",
    author_email="team@fractiverse.ai",
    description="A fractal-based AI system with recursive intelligence processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fractiverse/fractal-lib",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "plotly>=4.14.0",
        "matplotlib>=3.3.0",
        "dataclasses>=0.6",
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'isort>=5.0',
            'mypy>=0.900',
            'sphinx>=4.0',
        ],
    }
) 