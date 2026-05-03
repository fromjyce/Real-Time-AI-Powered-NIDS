from setuptools import setup, find_packages

setup(
    name="nids",
    version="1.0.0",
    packages=find_packages(where=".", include=["src", "src.*"]),
    python_requires=">=3.11",
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "nids-capture=src.capture.packet_capture:main",
            "nids-processor=src.processing.stream_processor:main",
            "nids-api=src.api.main:main",
            "nids-simulate=scripts.simulate_traffic:main",
            "nids-train=scripts.train_models:main",
        ]
    },
)
