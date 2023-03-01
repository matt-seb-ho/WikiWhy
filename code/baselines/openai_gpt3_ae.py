import json
import os
import time

import openai
import pandas as pd
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

TASK_EXPLANATION = "Generate an answer and a chain of reasoning that leads from the provided question to the answer\n\n"
TASK_EXAMPLES = "Question: Why did Hiroyuki Yamashita feel pressured writing \"Boruto: Naruto the Movie\"?\nAnswer: There were time constraints to writing \"Boruto: Naruto the Movie\"\nExplanation: Creativity is difficult when put on a strict timetable. There was a need to both produce a good movie and do so on a strict time budget. These two demands put stress on Hiroyuki Yamashita while he worked.\n\nQuestion: Why did Homer P. Rainey get fired by the University of Texas in 1944?\nAnswer: Homer P. Rainey had liberal views\nExplanation: If the University of Texas is conservative, they wouldn't want people working there who have liberal views.\n\nQuestion: Why are red maple buds which form in fall and winter often visible from a distance?\nAnswer: The large size and reddish tint of red maple buds\nExplanation: The color red stands out from a distance, so if the buds are red in the fall and winter, you'd be able to see them from a distance.\n\nQuestion: Why did the production costs of aluminum change in the late 20th century?\Answer: There were advances in technology, lower energy prices, a favorable exchange rate of the United States dollar, and lower alumina prices\nExplanation: With advances in technology, prices of manufacturing change usually because they are now easier and cheaper to make. In this case it is aluminum that the price changed on because the technology improved the process."

input_csv_path = "data/with_sm.csv"
df = pd.read_csv(input_csv_path, delimiter=",")
out = pd.DataFrame(columns=['id', 'explanation'])

for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=TASK_EXPLANATION+TASK_EXAMPLES+"Question: " +
            row['question'] + "\n",
            temperature=1,
            max_tokens=1024
        )
        out = pd.concat(
            [out, pd.DataFrame([{'id': row['id'], 'explanation': json.loads(str(response))["choices"][0]["text"].strip()}])])
        time.sleep(0.5)
    except Exception as e:
        print(e)
        break

out.to_csv('data/gpt_full_out.csv', index=False)
