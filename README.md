# PDF-QA

PDF-QA is a question-answering application that allows users to upload PDF documents and query them using natural language. The application leverages language models and vector databases to retrieve and answer questions based on the content of the uploaded PDFs.

**Link demo**: [demo](https://pdf-app-anm2gjckkrhgyxy4fgutcu.streamlit.app/)

## Features

- Upload multiple PDF files.
- Extract and process text from PDFs.
- Store processed text in a vector database.
- Query the vector database using llm.
- Retrieve and display answers based on the content of the PDFs.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/PDF-QA.git
    cd PDF-QA
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:

    Create a `.env` file in the root directory of the project and add your Cohere API key:

    ```env
    COHERE_API_KEY=your_cohere_api_key
    ```

## Usage

Run the application:

    ```bash
    streamlit run app.py
    ```
