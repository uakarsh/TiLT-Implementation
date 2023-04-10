from setuptools import setup, find_packages

setup(
  name = 'tilt_transformers',
  packages = find_packages(where="src"),
  package_dir = {"": "src", "docformer": "src/"},
  version = '0.1.0',
  license='MIT',
  description = 'Going Full-TILT Boogie on Document Understanding with Text-Image-Layout Transformer:',
  author = 'Akarsh Upadhay',
  author_email = 'akarshupadhyayabc@gmail.com',
  url = 'https://github.com/uakarsh/TiLT-Implementation',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'document understanding',
  ],
  install_requires=[
    'torch>=1.6',
    'torchvision',
    'transformers',
    'sentencepiece',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
  
)