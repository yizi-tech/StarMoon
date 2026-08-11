from setuptools import setup, find_packages

setup(
    name="starmoon-z1",
    version="0.1.0",
    description="StarMoon-z1: Small model training & inference framework (1B-14B)",
    author="Yizi Tech & Yunzhi Jiankang AI",
    packages=find_packages(),
    package_data={"StarMoonZ1": ["webui/*.html"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "sentencepiece>=0.1.99",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
    ],
    extras_require={
        "flash": ["flash-attn>=2.3.0"],
        "vllm": ["vllm>=0.3.0"],
        "llamacpp": ["llama-cpp-python>=0.2.0"],
        "server": ["fastapi>=0.104.0", "uvicorn>=0.24.0", "pydantic>=2.0.0"],
        "all": [
            "flash-attn>=2.3.0", "vllm>=0.3.0", "llama-cpp-python>=0.2.0",
            "fastapi>=0.104.0", "uvicorn>=0.24.0", "pydantic>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "starmoon-z1=StarMoonZ1.cli:main",
        ],
    },
)
