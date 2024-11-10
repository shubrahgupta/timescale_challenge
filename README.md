# timescale_challenge

*This is a submission for the [Open Source AI Challenge with pgai and Ollama ](https://dev.to/challenges/pgai)*

## What we Built
We (@khemraj_bawaskar_f283a984 and I) have developed a dynamic, Streamlit-based application that lets you dive into your codebase like never before! Imagine having a personal ChatGPT that not only understands your code but can answer your questions about it, providing insights and explanations straight from your files. Here’s what our app can do:

1. **Codebase Processor:** Automatically clones your GitHub repository, vectorizes your code, and stores these embeddings in the PostgreSQL Vector Database.
2. **Intelligent Chatbot**: Ready to answer all your code queries, it’s your code assistant, right at your fingertips!
Behind the scenes, it harnesses RAG (Retrieval-Augmented Generation) to embed important objects into a vector store, allowing for super-accurate similarity searches to give you the answers you need.

Built with: `langchain`, `psycopg2`
Languages: Python 🐍
<!-- Share an overview about your project. -->

## Demo
<!-- Share a link to your app and include some screenshots here. -->
The first view of the application looks like this:

![HomePage](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/jhkl3lluepwz1jinnj7u.png)

It has options in the left pane, to toggle between 'Process Codebase' and 'Chatbot', while the default selected option is 'Process Codebase'.

![Codebase processor in Action](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/e48936xxbwms4a7ztjdh.png)


On selecting Chatbot, we find this screen: 

![Chatbot Screen](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/072ay1ib3xqhec2gi9kf.png)
This screen provides a drop-down list to choose from the projects created, and get started on chatting with the assistant.

On asking queries, the response is delivered, and the screen looks something like this:

![Chatbot in Action](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/1r3nzujdphpe4pcu253m.png)







## Tools Used
<!-- Tell us how you used pgvector, pgvectorscale, pgai, and/or pgai Vectorizer -->

We have made use of: 

1. **pgvector**: We tapped into the open-source power of pgvector through the `langchain_postgres` library to store vectorized documents and handle similarity searches.
```
from langchain_postgres.vectorstores import PGVector
vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
        )
```


2. **pgai**: Our custom integration of Timescale's pgai with the LangChain's `PGVector` Class gave us a streamlined approach for creating embeddings! Using pgAI, we crafted a custom `PgAIEmbeddings` class to harness the OpenAI model via pgAI.
```
from langchain_postgres.vectorstores import PGVector
vectorstore = PGVector(
       embeddings=**PgAIEmbeddings**(connection_string=connection),
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
        )
```
Here’s a sneak peek at how we handled embeddings in our PgAIEmbeddings class:

```
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """Embed a list of documents using pgAI's OpenAI embedding function"""
    embeddings = []
    with psycopg2.connect(self.connection_string) as conn:
        with conn.cursor() as cur:
            for text in texts:
                cur.execute(
                    "SELECT ai.openai_embed('text-embedding-ada-002', %s) as embedding",
                    (text,)
                )
                result = cur.fetchone()
                # Parse the embedding string into a list of floats
                embedding = self._parse_embedding(result[0])
                embeddings.append(embedding)
    return embeddings
```
3. **Open AI**: Acting as both the embedding creator and the brains behind the answers, OpenAI’s LLM brings this platform to life!


## Final Thoughts
This platform is a complete game-changer for code exploration, making it fun, intuitive, and unbelievably insightful. Just imagine using it to create code summaries, suggest improvements, JIRA user story and test cases creation, modernize legacy code or even aid in debugging—possibilities are endless! With pgvector and pgai as our foundation, our app is set up for endless growth and flexibility.

Prize Categories:
We're in the running for the main prize category! This challenge asked us to build an AI application using open-source tools with PostgreSQL as a vector database, alongside at least two of the following: pgvector, pgvectorscale, pgai, and pgai Vectorizer. We did that—and more!

<!-- Team Submissions: Please pick one member to publish the submission and credit teammates by listing their DEV usernames directly in the body of the post. -->

<!-- Don't forget to add a cover image (if you want). -->

<!-- Thanks for participating! -->
