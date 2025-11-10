from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="groq-colab-agent",
    version="1.0.0",
    author="AI Systems Team",
    description="Enterprise-grade AI agent with Groq API integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/groq-colab-agent",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "groq>=0.4.1",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "aiofiles>=23.1.0",
        "sqlalchemy>=2.0.0",
        "requests>=2.31.0",
        "websockets>=12.0.0",
    ],
)
