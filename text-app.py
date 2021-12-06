import uvicorn
from fastapi import FastAPI  
from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_streaming_bulk
from elasticapm.contrib.starlette import make_apm_client, ElasticAPM
import spacy 
from transformers import pipeline

summarizer = pipeline(task="summarization", model="sshleifer/distilbart-cnn-6-6", tokenizer="sshleifer/distilbart-cnn-6-6",framework="pt")

def summarize(text):
    summary = summarizer(text, max_length=40, min_length=20, do_sample=False)
    return summary


nlp = spacy.load("en_core_web_md")


def ner_spacy(text):
    doc1 = nlp(text)
    for ent in doc1.ents:
        yield ent.text, ent.label_

apm = make_apm_client(
    {"SERVICE_NAME": "text-app", "SERVER_URL": "http://localhost:8200"}
)
# es = AsyncElasticsearch(os.environ["ELASTICSEARCH_HOSTS"])
es = AsyncElasticsearch()
app = FastAPI(
    title = "NEW API",
    description = "NER extracts labels from the text"
)

app.add_middleware(ElasticAPM, client=apm)



@app.on_event("shutdown")
async def app_shutdown():
    await es.close()


@app.get('/ner')
def ner_text(text: str):
    output_spacy = list(ner_spacy(text))
    result = {"NER from Spacy": output_spacy}
    return result

@app.get('/summary')
def summarize_text(text: str):
    summary = summarize(text)
    result = {"Summary from transformers": summary}
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
