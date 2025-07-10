from fastapi import APIRouter, UploadFile, File
from tempfile import NamedTemporaryFile
import pdf4llm
from fastapi.responses import JSONResponse
from .supportings import indexing_documents, json_graph_data, get_retrived_context, generated_answer
from .validation import Query, Metrics, exact_match, f1_score, Generation

router = APIRouter(tags=["routes"], prefix="/rag")


@router.post("/indexing")
async def indexing(documents: list[UploadFile] = File(...)):

    for myfile in documents:
        read_file = await myfile.read()

        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(read_file)
            tempname = tmp.name
        try:
            data = pdf4llm.to_markdown(tempname, page_chunks=True)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        graph = await indexing_documents(data)


        json_data = await json_graph_data(graph)

    return JSONResponse({"knowledge_graph": json_data})



@router.post("/contextgenreration")
async def query(request: Query):

    refined_query = request.query.strip()
    if refined_query:
        retrived_context = await get_retrived_context(refined_query, request.knowledge_graph)
    else:
        retrived_context = None

    return JSONResponse({"query": refined_query,"context": retrived_context})


@router.post("/generate")
async def generate(request: Generation):

        final_answer = await generated_answer(request.query, request.context.strip())

        return JSONResponse({"answer": final_answer})


@router.post("/evaluation")
async def evaluate(request: Metrics):
    em_score = await exact_match(request.prediction, request.truth)
    f1score = await f1_score(request.prediction, request.truth)
    return JSONResponse({"em_score": em_score, "f1score": f1score})
    _






