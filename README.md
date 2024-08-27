# LLM Assistant

This repository contains a chatbot application. The chatbot uses various tools to provide responses based on user input.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)

## Installation

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Set up environment variables:**

    Create a `.env` file in the root directory of the project and add the necessary environment variables:
    ```sh
    APP_ENV=development
    GRADIO_USERNAME=<your-username>
    GRADIO_PASSWORD=<your-password>
    ```

2. **Run the application:**
    ```sh
    python app.py --config=config.yaml
    ```

## Configuration

The application requires a configuration file (`config.yaml`) to run. This file should contain the various configuration options needed for the app. Below is an example configuration:
```yaml
title: "RAG Assistant"
# Add other configuration options as needed
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/license/mit) file for more details.
