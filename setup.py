from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(
  name="realestate_vss",
  version="0.0.1",
  author="Kelvin Chan",
  author_email="kechan.ca@gmail.com",
  description="Real Estate Vector Similarity Search",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/kechan/realestate-vss",
  packages=find_packages(),
)