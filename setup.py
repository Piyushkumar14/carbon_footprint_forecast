from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -

REPO_NAME = "carbon_footprint_forecast"
AUTHOR_USER_NAME = "Piyushkumar14"
SRC_REPO = "src"


setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="A small package for forecasting carbon footprint.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="pr14122001@gmail.com",
    packages=[SRC_REPO],
    license="MIT",
    python_requires=">=3.7",
)