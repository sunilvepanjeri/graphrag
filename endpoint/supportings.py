from sentence_transformers import SentenceTransformer
import networkx as nx
from networkx.readwrite import json_graph
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI



client = OpenAI()


import json
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

graph = nx.Graph()


async def indexing_documents(documents):
    final_embeddings = []
    nodes = 0
    for document in documents:
        text = document["text"]
        embedding = await embed_query(text)
        final_embeddings.append(embedding)
        graph.add_node(nodes, text = text)
        nodes += 1
    for i in range(len(final_embeddings)):
        for j in range(i + 1, len(final_embeddings)):
            similarity = await similarity_output(final_embeddings[i], final_embeddings[j])
            if similarity > 0.5:
                graph.add_edge(i, j, weight = similarity)
    return graph


async def json_graph_data(graph):
    data = json_graph.node_link_data(graph)
    json_data = json.dumps(data)
    return json_data


async def embed_query(query: str):
    embedding = model.encode(query)
    return embedding.reshape(-1, 1)



async def similarity_output(a, b):
    return int(cosine_similarity(a,b)[0][0])


async def get_retrived_context(query: str, knowledge: str):
    knowledge = json.loads(knowledge)

    context_graph = json_graph.node_link_graph(knowledge)

    query_embedding = await embed_query(query)

    query_matching = [(node, cosine_similarity(query_embedding, model.encode(context_graph.nodes[node]["text"]).reshape(-1, 1))[0][0])
                     for node in context_graph.nodes]

    top_query = sorted(query_matching,key = lambda x: (x[1], x[0]), reverse = True)[0]

    top_context = context_graph.nodes[top_query[0]]["text"]

    edge_list_for_top_query = dict(dict(context_graph.adj)[top_query[0]])

    edge_context = [key for key in edge_list_for_top_query if edge_list_for_top_query[key]['weight'] > 0.8][ : 2]


    final_context = top_context + "".join([context_graph.nodes[i]["text"] for i in edge_context])

    return final_context


async def generated_answer(query: str, context: str):
    input_text = f"question: {query} context: {context}"

    response = client.responses.create(
        model="gpt-4.1",
        input=input_text
    )

    return response.output_text







