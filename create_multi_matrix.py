import openai
import random
import numpy as np
from tqdm import tqdm


from openai import OpenAI

client = OpenAI(api_key=)


def generate_number(digits):
    return random.randint(10**(digits-1), 10**digits -1)


def generate_addition_problem(d1, d2):
    a = generate_number(d1)
    b = generate_number(d2)
    return a, b, a*b

def query_model(prompt):
    try :
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": prompt}],
            #temperature=0.0
        )
        #print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("API error:", str(e))
        return ""
    



def test_accuracy_for_pair(d1, d2, n_samples=50):
    correct=0
    for _ in range (n_samples):
        a, b, true_answer = generate_addition_problem(d1, d2)
        response = query_model (f"What is the result for {a} times {b}? Give your answer as a single number. I want you to only ouput a single number without anything else. Furthermore, include no comma or punctuation in the output.")
        try:
            predicted = int(response)
            if predicted == true_answer:
                correct += 1
        except:
            continue
    print("There are this many correct:", correct)
    return (correct/n_samples)*100


accuracy_matrix = np.zeros((20, 20))

for i in tqdm(range(20), desc="Digit 1"):
    for j in range(20):
        acc = test_accuracy_for_pair(i+1, j+1, n_samples=3)
        accuracy_matrix[i][j] = acc
        #print("j", j)
    #print("i", i)


print(accuracy_matrix)
np.save("accuracy_matrix_gpt4o.npy", accuracy_matrix)





