"""
OpenBroadcast — Package Setup

Build with:
    python setup.py build

Create .exe with PyInstaller:
    pyinstaller --onefile --name OpenBroadcast main.py
"""

from setuptools import setup, find_packages

setup(
    name="openbroadcast",
    version="1.0.0",
    author="OpenBroadcast Team",
    description="Eye gaze correction for video calls — works on low-end PCs without GPU",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "PyQt6>=6.5.0",
        "psutil>=5.9.0",
        "py-cpuinfo>=9.0.0",
        "onnxruntime>=1.15.0",
    ],
    extras_require={
        "gpu": ["torch>=2.0.0"],
        "virtualcam": ["pyvirtualcam>=0.4.0"],
        "wmi": ["WMI>=1.5.1"],
        "display": ["screeninfo>=0.8.1"],
        "training": ["torch>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "openbroadcast=openbroadcast.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
