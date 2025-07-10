from pydantic import BaseModel

class Query(BaseModel):
    query: str
    knowledge_graph: str


class Metrics(BaseModel):
    prediction: str
    truth: str

class Generation(BaseModel):
    query: str
    context: str


async def exact_match(pred, truth):
    return int(pred.strip().lower() == truth.strip().lower())

async def f1_score(pred, truth):
    pred_tokens = set(pred.lower().split())
    truth_tokens = set(truth.lower().split())
    common = pred_tokens & truth_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)