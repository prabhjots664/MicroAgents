from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="microAgents",
    version="1.3.7",
    author="MicroAgents Team",
    author_email="prabhjots664@gmail.com",
    description="A lightweight LLM orchestration framework for building Multi-Agent AI systems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prabhjots664/MicroAgents",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "requests>=2.25.0",
        "urllib3>=1.26.0"
    ],
)
