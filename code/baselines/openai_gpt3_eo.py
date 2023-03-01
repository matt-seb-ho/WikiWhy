import json
import os
import time

import openai
import pandas as pd
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

TASK_EXPLANATION = "Generate a chain of reasoning that leads from the cause to the effect\n\n"
TASK_EXAMPLES = "Cause: There were time constraints to writing \"Boruto: Naruto the Movie\"\nEffect: Hiroyuki Yamashita felt pressured writing \"Boruto: Naruto the Movie\"\nExplanation: Creativity is difficult when put on a strict timetable. There was a need to both produce a good movie and do so on a strict time budget. These two demands put stress on Hiroyuki Yamashita while he worked.\n\nCause: Homer P. Rainey had liberal views. \nEffect: Homer P. Rainey was fired by the University of Texas in 1944.\nExplanation: If the University of Texas is conservative, they wouldn't want people working there who have liberal views.\n\nCause: the large size and reddish tint of red maple buds\nEffect: Red maple buds which form in fall and winter are often visible from a distance.\nExplanation: The color red stands out from a distance, so if the buds are red in the fall and winter, you'd be able to see them from a distance.\n\nCause: There were advances in technology, lower energy prices, a favorable exchange rate of the United States dollar, and lower alumina prices. \nEffect: Productions costs of aluminum changed in the late 20th century.\nExplanation: With advances in technology, prices of manufacturing change usually because they are now easier and cheaper to make. In this case it is aluminum that the price changed on because the technology improved the process.\n\n"

input_csv_path = "iclr/test_set_updated.csv"
df = pd.read_csv(input_csv_path, delimiter=",")
out = pd.DataFrame(columns=['id', 'explanation'])

for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    prompt = TASK_EXPLANATION + TASK_EXAMPLES + "Cause: " + row['cause'] + "\nEffect: " + row['effect'] + "\nExplanation: "
    try:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=0,
            max_tokens=1024
        )
        out = pd.concat(
            [out, pd.DataFrame([{'id': row['id'], 'explanation': json.loads(str(response))["choices"][0]["text"].strip()}])])
        time.sleep(0.5)
    except:
        break

out.to_csv('data/gpt_iclr_out.csv', index=False)
