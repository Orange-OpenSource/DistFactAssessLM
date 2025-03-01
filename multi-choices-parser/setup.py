from setuptools import setup, find_packages

setup(
    name='multi-choices-parser',
    version='0.9.57',
    author='Hichem Ammar Khodja',
    author_email='hichem5696@gmail.com',
    packages=find_packages(),
    description='An efficient incremental parser for multi-choices grammars.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/HichemAK/multi-choices-parser',
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ]
)
