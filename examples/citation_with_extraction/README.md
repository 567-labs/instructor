# Citation with Extraction

This repository contains a FastAPI application that uses GPT-4 to answer questions based on a given context and extract relevant facts with correct and exact citations. The extracted facts are returned as JSON events using Server-Sent Events (SSE).

## How it Works

The FastAPI app defines an endpoint `/extract` that accepts a POST request with JSON data containing a `context` and a `query`. The `context` represents the text from which the question is being asked, and the `query` is the question itself.

The app leverages GPT-4, an advanced language model, to generate answers to the questions and extract relevant facts. It ensures that the extracted facts include direct quotes from the given context.

## Example Usage

To use the `/extract` endpoint, send a POST request with `curl` or any HTTP client with the following format:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "context": "My name is Jason Liu, and I grew up in Toronto Canada but I was born in China.I went to an arts highschool but in university I studied Computational Mathematics and physics.  As part of coop I worked at many companies including Stitchfix, Facebook.  I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.",
  "query": "What did the author do in school?"
}' -N http://localhost:8000/extract
```

```sh
data: {'body': 'In school, the author went to an arts high school.', 'spans': [(91, 106)], 'citation': ['arts highschool']}
data: {'body': 'In university, the author studied Computational Mathematics and physics.', 'spans': [(135, 172)], 'citation': ['Computational Mathematics and physics']}
```

Replace `http://localhost:8000` with the actual URL of your FastAPI app if it's running on a different host and port. The API will respond with Server-Sent Events (SSE) containing the extracted facts in real-time.

## Bring your own API key

If you have your own api key but dont want to try deploying it yourself you're welcome to use my 
modal instance here, this code is public and I do not store your key.

```bash
curl -X 'POST' \
  'https://jxnl--rag-citation-fastapi-app.modal.run/extract' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <OPENAI_API_KEY>' \
  -d '{
  "context": "My name is Jason Liu, and I grew up in Toronto Canada but I was born in China.I went to an arts highschool but in university I studied Computational Mathematics and physics.  As part of coop I worked at many companies including Stitchfix, Facebook.  I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.",
  "query": "What did the author do in school?"
}'
```


## Requirements

To run this application, ensure you have the following Python packages installed:

```bash
pip install -r requirements.txt
```

## Running the App

To run the FastAPI app, execute the following command:

```bash
uvicorn main:app --reload
```

This will start the server, and the `/extract` endpoint will be available at `http://localhost:8000/extract`.

## Note

Ensure that you have a valid API key for GPT-4 from OpenAI. If you don't have one, you can obtain it from the OpenAI website.

Please use this application responsibly and be mindful of any usage limits or restrictions from OpenAI's API usage policy.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as you see fit.