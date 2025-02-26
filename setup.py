from setuptools import setup, find_packages

setup(
    name="fractiverse",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "networkx>=3.0",
        "psutil>=5.9.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "websockets>=11.0.0",
        "pycryptodomex>=3.19.0",
        "structlog>=23.1.0",
        "prometheus-client>=0.17.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-sugar>=1.0.0",
            "pytest-xdist>=3.6.0",
            "pytest-timeout>=2.3.0",
            "pytest-benchmark>=5.1.0",
            "httpx>=0.24.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0"
        ]
    },
    python_requires=">=3.10",
    author="FractiVerse Team",
    author_email="team@fractiverse.ai",
    description="A fractal intelligence system with cognitive processing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fractiverse/fractiverse",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 