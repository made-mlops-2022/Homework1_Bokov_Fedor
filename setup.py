from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='MADE MLOps HW1',
    author='Fedor Bokov',
    license='BSD-3',
    python_requires="<=3.9.12",
    install_requires=[
        "click",
        "Sphinx",
        'coverage',
        'awscli',
        'flake8',
        'pandas',
        'numpy',
        'sweetviz',
        'argparse',
        'logging',
        'catboost',
        'sklearn',
        'pytest'
    ],
)
