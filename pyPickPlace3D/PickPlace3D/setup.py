import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PickPlace3D",
    version="0.0.1.1",
    author="Ben Knight",
    author_email="bknight@i3drobotics.com",
    description="Pick and place using 3D data to inform robotic movement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/i3drobotics/PickPlace3D",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy','opencv-python','stereocapture'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)