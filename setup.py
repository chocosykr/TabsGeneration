from setuptools import setup, find_packages

setup(
    name="ukulele-tabs",
    version="0.1.0",
    description="Differentiable Ukulele Tab Transcription Pipeline",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "librosa",
        "numpy",
        "scipy",
        "tqdm"
    ],
    entry_points={
        "console_scripts": [
            "utab=prototype.cli:main",
        ]
    },
)
