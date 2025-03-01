from setuptools import setup, find_packages

setup(
    name="fractiverse-core",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy<2.0.0",
        "scipy>=1.7.0",
    ],
    author="FractiVerse Team",
    description="FractiVerse Core - Quantum-Fractal Cognitive System",
) 