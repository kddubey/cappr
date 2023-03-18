from setuptools import setup, find_packages


with open('README.md', mode='r', encoding='utf-8') as f:
    readme = f.read()


setup(name='lm-classification',
      version='1.0',
      description='Zero-shot text classification using OpenAI language models',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/kddubey/lm-classification/',
      license='MIT',
      python_requires='>=3.8.0',
      install_requires=['numpy>=1.21.0',
                        'openai>=0.26.0',
                        'tiktoken>=0.2.0',
                        'tqdm>=4.27.0',
                        'torch>=1.12.1', ## TODO: split huggingface vs openai
                        'transformers>=4.26.1'],
      author_email='kushdubey63@gmail.com',
      packages=find_packages())
