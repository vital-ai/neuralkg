from setuptools import setup, find_packages

setup(
    name='vital-neuralkg',
    version='0.0.1',
    author='Marc Hadfield',
    author_email='marc@vital.ai',
    description='Vital NeuralKG',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vital-ai/neuralkg',
    packages=find_packages(exclude=["test", "test_data"]),
    license='Apache License 2.0',
    install_requires=[

        'klaycircuits',
        'numpy'

    ],
    extras_require={

    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
