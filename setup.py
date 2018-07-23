import setuptools


setuptools.setup(
    name="itemCheck",
    version="0.0.1",
    author="Yadong Sun",
    author_email="xxxspy@126.com",
    description="A simple python wrapper for matter.js",
    long_description='long_description',
    long_description_content_type="text/markdown",
    url="https://github.com/xxxspy/matter.py",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ),
    package_data={
            '.': ['template.html'],
    }
)