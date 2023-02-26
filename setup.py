from setuptools import setup, find_packages


with open('README.md', mode='r', encoding='utf-8') as f:
    readme = f.read()


setup(name='lm-classification',
      version='1.0',
      description='Zero-shot text classification using OpenAI language models',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/kddubey/lm-classification/',
      license='GNU General Public License v3.0',
      python_requires='>=3.8.0',
      install_requires=['numpy>=1.17.3',
                        'openai>=0.26.0',
                        'tqdm>=4.27.0',
                        'transformers>=3.2.0'],
      author_email='kushdubey63@gmail.com',
      packages=find_packages())
