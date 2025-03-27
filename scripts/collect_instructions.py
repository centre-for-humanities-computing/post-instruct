import ast
import json
import os
import random
from pathlib import Path

import openai
from tqdm import tqdm

from utils import load_tasks

example_instructions = {
    "STS": ["Retrieve semantically similar text."],
    "Summarization": [
        "Given a news summary, retrieve other semantically similar summaries"
    ],
    "BitextMining": "Retrieve parallel sentences.",
    "Classification": [
        "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
        "Classify Amazon reviews into positive or negative sentiment",
        "Classify the given Amazon review into its appropriate rating category",
        "Given a online banking query, find the corresponding intents",
        "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
        "Classify the sentiment expressed in the given movie review text from the IMDB dataset",
        "Given a user utterance as query, find the user intents",
        "Given a user utterance as query, find the user scenarios",
        "Classify the intent domain of the given utterance in task-oriented conversation",
        "Classify the intent of the given utterance in task-oriented conversation",
        "Classify the given comments as either toxic or not toxic",
        "Classify the sentiment of a given tweet as either positive, negative, or neutral",
        "Classify the fine-grained category of the given news title",
        "Given an App description text, find the appropriate fine-grained category",
        "Classify sentiment of the customer review into positive, neutral, or negative",
        "Classify the customer review for iPhone on e-commerce platform into positive or negative",
        "Classify the customer review for online shopping into positive or negative",
        "Classify the customer review from a food takeaway platform into positive or negative",
    ],
    "Clustering": [
        "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
        "Identify the main and secondary category of Arxiv papers based on the titles",
        "Identify the main category of Biorxiv papers based on the titles and abstracts",
        "Identify the main category of Biorxiv papers based on the titles",
        "Identify the main category of Medrxiv papers based on the titles and abstracts",
        "Identify the main category of Medrxiv papers based on the titles",
        "Identify the topic or theme of Reddit posts based on the titles",
        "Identify the topic or theme of Reddit posts based on the titles and posts",
        "Identify the topic or theme of StackExchange posts based on the titles",
        "Identify the topic or theme of StackExchange posts based on the given paragraphs",
        "Identify the topic or theme of the given news articles",
        "Identify the main category of scholar papers based on the titles",
        "Identify the main category of scholar papers based on the titles and abstracts",
        "Identify the topic or theme of the given news articles based on the titles",
        "Identify the topic or theme of the given news articles based on the titles and contents",
    ],
    "Reranking": [
        "Retrieve duplicate questions from AskUbuntu forum",
        "Retrieve relevant news articles based on user browsing history",
        "Given a title of a scientific paper, retrieve the titles of other relevant papers",
        "Retrieve duplicate questions from StackOverflow forum",
        "Retrieve duplicate questions from Sprint forum",
        "Retrieve tweets that are semantically similar to the given tweet",
        "Retrieve tweets that are semantically similar to the given tweet",
        "Given a Chinese search query, retrieve web passages that answer the question",
        "Given a Chinese search query, retrieve web passages that answer the question",
        "Given a Chinese community medical question, retrieve replies that best answer the question",
        "Given a Chinese community medical question, retrieve replies that best answer the question",
        "Retrieve semantically similar text.",
        "Retrieve semantically similar text.",
    ],
    "PairClassification": [
        "Retrieve duplicate questions from AskUbuntu forum",
        "Retrieve relevant news articles based on user browsing history",
        "Given a title of a scientific paper, retrieve the titles of other relevant papers",
        "Retrieve duplicate questions from StackOverflow forum",
        "Retrieve duplicate questions from Sprint forum",
        "Retrieve tweets that are semantically similar to the given tweet",
        "Retrieve tweets that are semantically similar to the given tweet",
        "Given a Chinese search query, retrieve web passages that answer the question",
        "Given a Chinese search query, retrieve web passages that answer the question",
        "Given a Chinese community medical question, retrieve replies that best answer the question",
        "Given a Chinese community medical question, retrieve replies that best answer the question",
        "Retrieve semantically similar text.",
        "Retrieve semantically similar text.",
    ],
    "Retrieval": [
        "Given a claim, find documents that refute the claim",
        "Given a claim about climate change, retrieve documents that support or refute the claim",
        "Given a query, retrieve relevant entity descriptions from DBPedia",
        "Given a claim, retrieve documents that support or refute the claim",
        "Given a financial question, retrieve user replies that best answer the question",
        "Given a multi-hop question, retrieve documents that can help answer the question",
        "Given a web search query, retrieve relevant passages that answer the query",
        "Given a question, retrieve relevant documents that best answer the question",
        "Given a question, retrieve Wikipedia passages that answer the question",
        "Given a question, retrieve questions that are semantically equivalent to the given question",
        "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
        "Given a scientific claim, retrieve documents that support or refute the claim",
        "Given a question, retrieve detailed and persuasive arguments that answer the question",
        "Given a query on COVID-19, retrieve documents that answer the query",
        "Given a Chinese search query, retrieve web passages that answer the question",
        "Given a web search query, retrieve relevant passages that answer the query",
        "Given a Chinese search query, retrieve web passages that answer the question",
        "Given a question on COVID-19, retrieve news articles that answer the question",
        "Given a Chinese community medical question, retrieve replies that best answer the question",
        "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
        "Given a medical question, retrieve user replies that best answer the question",
        "Given a video search query, retrieve the titles of relevant videos",
    ],
}

prompt_template = """
I need to use an embedding model for retrieval. The embedding model takes instructions as to what task it has to complete.

Here are some examples of instructions it can use:
{examples}

Here is a short descriptions of the dataset that I need to use this model for:
'''{description}'''

The dataset contains the following languages: {languages} and domains: {domains}, and is of task type: {task_subtypes}

Please generate 10 possible instructions for this dataset in form of a Python list. Respond with nothing else but the list of instructions.
"""


def generate_instructions(task, client) -> list[str]:
    _instructions = example_instructions.get(task.metadata.type, [])
    if not _instructions:
        raise ValueError(
            "Couldn't find example instructions for the task type at hand."
        )
    examples = " - " + "\n - ".join(random.sample(_instructions, 8))
    description = task.metadata.description
    languages = ", ".join(task.metadata.languages)
    domains = ", ".join(task.metadata.domains)
    task_subtypes = ", ".join(task.metadata.task_subtypes)
    prompt = prompt_template.format(
        examples=examples,
        description=description,
        languages=languages,
        domains=domains,
        task_subtypes=task_subtypes,
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        max_tokens=256,
    )
    response_str = response.choices[0].message.content
    return ast.literal_eval("[" + response_str.split("[")[-1].split("]")[0] + "]")


def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=openai.api_key)
    random.seed(42)
    tasks = load_tasks()
    out_path = Path("dat/synthetic_instructions.json")
    out_path.parent.mkdir(exist_ok=True)
    instructions = {}
    for task in tqdm(tasks, desc="Generating instructions for all tasks."):
        try:
            instructions[task.metadata.name] = generate_instructions(task, client)
        except Exception as e:
            print(
                f"[WARNING] Couldn't generate instructions for task {task.metadata.name} due to {e}"
            )
    with out_path.open("w") as out_file:
        out_file.write(json.dumps(instructions))
    print("DONE")


if __name__ == "__main__":
    main()
