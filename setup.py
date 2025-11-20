from setuptools import setup, find_packages

setup(
    name="rinex_spoofing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xarray",
        "georinex",
        "matplotlib",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "rinex-to-json=rinex_spoofing.rinex_to_json:cli",
            "spoof-detection=rinex_spoofing.spoof_detection:main",
            "visualize-nav=rinex_spoofing.visualize_nav:main",
        ],
    },
)
