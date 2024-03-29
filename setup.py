import os
from setuptools import setup, find_packages


with open(os.path.join("src", "cappr", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__ = "):
            version = str(line.split()[-1].strip('"'))
            break


requirements = [
    "numpy>=1.21.0",
    "tqdm>=4.27.0",
]
requirements_openai = [
    "openai>=0.26.0",
    "tiktoken>=0.2.0",
]
requirements_huggingface = [
    "transformers[torch]>=4.31.0",
]
requirements_huggingface_dev = [
    "transformers[torch]>=4.35.0",  # to test AutoGPTQ on CPU and AutoAWQ with caching
    "huggingface-hub>=0.16.4",
    "sentencepiece>=0.1.99",
]
requirements_llama_cpp = ["llama-cpp-python>=0.2.11"]
requirements_llama_cpp_dev = ["llama-cpp-python>=0.2.13"]  # to test Bloom
requirements_demos = [
    "datasets>=2.10.0",
    "jupyter>=1.0.0",
    "matplotlib>=3.7.3",
    "pandas>=1.5.3",
    "scikit-learn>=1.2.2",
]
requirements_dev = [
    "docutils<0.19",
    "pre-commit>=3.5.0",
    "pydata-sphinx-theme>=0.13.1",
    "pytest>=7.2.1",
    "pytest-cov>=4.0.0",
    "ruff>=0.3.0",
    "sphinx>=6.1.3",
    "sphinx-copybutton>=0.5.2",
    "sphinx-togglebutton>=0.3.2",
    "sphinxcontrib-napoleon>=0.7",
    "twine>=4.0.2",
]


with open("README.md", mode="r", encoding="utf-8") as f:
    readme = f.read()


setup(
    name="cappr",
    version=version,
    description="Completion After Prompt Probability. Make your LLM make a choice",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/kddubey/cappr/",
    license="Apache License 2.0",
    python_requires=">=3.8.0",
    install_requires=requirements,
    extras_require={
        "openai": requirements_openai,
        "hf": requirements_huggingface,
        "llama-cpp": requirements_llama_cpp,
        "all": requirements_openai + requirements_huggingface + requirements_llama_cpp,
        "demos": (
            requirements_openai
            + requirements_huggingface_dev
            + requirements_llama_cpp_dev
            + requirements_demos
        ),
        "dev": (
            requirements_openai
            + requirements_huggingface_dev
            + requirements_llama_cpp_dev
            + requirements_demos
            + requirements_dev
        ),
    },
    author_email="kushdubey63@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
