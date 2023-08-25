from setuptools import setup

setup(
    name="bias-bench",
    version="0.1.0",
    description="An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models",
    url="https://github.com/mcgill-nlp/bias-bench",
    packages=["bias_bench"],
    install_requires=[
        "torch",
        "transformers",
        "scipy",
        "scikit-learn",
        "nltk",
        "datasets",
        "accelerate",
    ],
    include_package_data=True,
    zip_safe=False,
)
