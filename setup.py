import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="simple_esn",
    version="1.0",
    author="Sylvain Chevallier",
    author_email="sylvain.chevallier@uvsq.fr",
    description="Simple Echo State Network within sklearn framework",
    url="https://github.com/sylvchev/simple_esn",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=requirements
)
