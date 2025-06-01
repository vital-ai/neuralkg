from setuptools import setup, find_packages

setup(
    name='vital-neuralkg',
    version='0.0.2',
    author='Marc Hadfield',
    author_email='marc@vital.ai',
    description='Vital NeuralKG',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vital-ai/neuralkg',
    packages=find_packages(exclude=["test", "test_data"]),
    license='Apache License 2.0',
    install_requires=[

        'vital-ai-vitalsigns>=0.1.27',
        'vital-ai-domain>=0.1.4',
        'vital-ai-haley-kg>=0.1.24',
        'kgraphservice>=0.0.6',

        'pykeen',
        'torch',
        'problog',
        'pysdd',

        'klaycircuits>=0.0.2',
        'numpy',

        'scallopy',
        'lark>=1.2.2'

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
