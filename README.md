## ML Eng Tutorials

These are the scripts used for the course [ML Engineering for Production](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops).
The scripts are based off the course Jupyter notebooks, but structured for a production environment

### Functionality
- A basic object detection model, deployed using `fastAPI`

## Installation
*This requires Python 3.9.6. You can use another version, provided you update the requirement in the `pyproject.toml`*

This repo uses a poetry environment. Once you have poetry installed,

```bash
cd ml_eng_tutorials

# initiate the poetry shell
poetry shell

# install packages
poetry install
```
I've manually installed `tensorflow` using `python -m pip install tensorflow`

The images in this repo are sourced from [here](https://github.com/https-deeplearning-ai/machine-learning-engineering-for-production-public/tree/main/course1/week1-ungraded-lab/images)

## Running the app
```bash
cd src/app

# start the server locally
uvicorn main:app --reload

# then load http://localhost:8000/docs and follow the instructions on the screen
```