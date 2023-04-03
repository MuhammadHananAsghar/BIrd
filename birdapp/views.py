from django.shortcuts import render
from IPython.display import Markdown
from django.views import View
from dotenv import dotenv_values
import pinecone
import markdown
import openai


config = dotenv_values()

index_name = config['INDEX_NAME']
embed_model = config['EMBED_MODEL']

openai.api_key = config['OPENAI_API_KEY']
# initialize connection to pinecone
pinecone.init(
    api_key=config['PINECONE_API_KEY'],  # app.pinecone.io (console)
    environment=config['ENVIROMENT']  # next to API key in console
)
# connect to index
index = pinecone.GRPCIndex(index_name)


def query(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts (including the questions)
    res = index.query(xq, top_k=5, include_metadata=True)
    contexts = [item['metadata']['text'] for item in res['matches']]
    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

    # system message to 'prime' the model
    primer = f"""You are Q&A bot. A highly intelligent system that answers
    user questions based on the information provided by the user above
    each question. If the information can not be found in the information
    provided by the user you truthfully say "I don't know". Give Answers in proper paragraphs with proper bolds so that i can render in html textarea.
    """

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query}
        ]
    )

    return res

class Index(View):
    def get(self, request, *args, **kwargs):
        search_query = request.GET.get("query", None)
        displayAble = ''
        if search_query != None:
            response = query(search_query)
            displayAble = response['choices'][0]['message']['content']
        # displayAble = Markdown(response['choices'][0]['message']['content'])
        return render(request, 'index.html', {'displayAble': displayAble})