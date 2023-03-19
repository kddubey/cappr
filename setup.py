from setuptools import setup, find_packages


requirements_base = [
    'numpy>=1.21.0',
    'tqdm>=4.27.0',
]

requirements_openai = [
    'openai>=0.26.0',
    'tiktoken>=0.2.0',
]

requirements_huggingface = [
    'torch>=1.12.1',
    'transformers>=4.26.1',
]

requirements_demos = [
    'datasets>=2.10.0',
    'jupyter>=1.0.0',
    'pandas>=1.5.3',
]

requirements_dev = [
    'pytest>=7.2.1',
]


with open('README.md', mode='r', encoding='utf-8') as f:
    readme = f.read()


setup(name='callm',
      version='1.0',
      description='Zero-shot text classification using OpenAI language models',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/kddubey/callm/',
      license='Apache License 2.0',
      python_requires='>=3.8.0',
      install_requires=requirements_base + requirements_openai,
      extras_requires = {'hf': requirements_huggingface,
                         'demos': requirements_huggingface + requirements_demos,
                         'dev': (requirements_huggingface +
                                 requirements_demos + requirements_dev),
                        },
      author_email='kushdubey63@gmail.com',
      packages=find_packages())
