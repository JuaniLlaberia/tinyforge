# Domain eval: base vs SFT

Greedy decoding. Base answers continue the completion prompt; SFT answers reply to the
`chat_prompt` through the chat template. Metrics use generated tokens only.

| Tag | Mode | Max new tokens | Stop rate | Repeated-4-gram rate, first 60 tokens | Repeated-4-gram rate, whole answer | Mean answer tokens |
| --- | ---- | -------------- | --------- | --- | --- | ------------------ |
| `base` | completion | 60 | 0/15 | 0.63 | 0.63 | 60 |
| `sft-2k` | chat | 256 | 0/15 | 0.28 | 0.75 | 256 |
| `sft-50k` | chat | 256 | 5/15 | 0.09 | 0.32 | 197 |

## On-topic (manual)

_y if the answer addresses the question or subject asked, regardless of correctness or repetition; n if it drifts, misreads the subject, or never attempts an answer_

| Prompt | `base` | `sft-2k` | `sft-50k` |
| ------ | --- | --- | --- |
| general-01 | n | y | y |
| general-02 | y | y | y |
| general-03 | y | y | n |
| history-01 | y | y | y |
| reasoning-01 | y | y | y |
| reasoning-02 | y | n | n |
| reasoning-03 | y | n | n |
| code-01 | n | y | y |
| code-02 | n | y | y |
| code-03 | n | y | y |
| writing-01 | y | n | y |
| writing-02 | y | y | y |
| writing-03 | y | y | y |
| research-01 | n | n | n |
| research-02 | y | y | y |
| **Total y** | 10/15 | 11/15 | 11/15 |

## Correct (manual)

_y if the core answer is right (code: the function itself works); n otherwise; '-' for writing prompts, which have no single right answer_

| Prompt | `base` | `sft-2k` | `sft-50k` |
| ------ | --- | --- | --- |
| general-01 | n | y | y |
| general-02 | n | n | y |
| general-03 | y | n | n |
| history-01 | n | n | n |
| reasoning-01 | n | n | n |
| reasoning-02 | n | n | n |
| reasoning-03 | n | n | n |
| code-01 | n | n | n |
| code-02 | n | n | n |
| code-03 | n | n | y |
| writing-01 | - | - | - |
| writing-02 | - | - | - |
| writing-03 | - | - | - |
| research-01 | n | n | n |
| research-02 | n | n | n |
| **Total y** | 1/12 | 1/12 | 3/12 |

Judgments: first-pass draft, to be reviewed (`eval/results/domain-judgments.json`).

## Answers

### general-01 (general_knowledge)

Completion prompt: `'The capital of France is'` · Chat prompt: "What is the capital of France?"

**`base`** (60 tokens, hit limit, repetition 0.14)

````text
the capital of the French Republic. The capital of the United States is Washington, D.C. The capital of the United States is Philadelphia, Pennsylvania.
The United States is the largest country in the world. It is the second largest country in the world. The United
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.92)

````text
The capital of France is the city of Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris.

The capital of France is Paris. The capital of France is Paris.

The capital of France is Paris. The capital of France is Paris.

The capital of France is Paris. The capital of France is Paris.

The capital of France is Paris. The capital of France is Paris.
 [...]
````

**`sft-50k`** (54 tokens, stopped, repetition 0.04)

````text
The capital of France is Paris. Paris is renowned for its historical landmarks, cultural institutions, and intellectual property. It is home to the iconic Eiffel Tower, the Louvre Museum, and the Louvre Museum.
````

### general-02 (general_knowledge)

Completion prompt: `'The largest planet in the solar system is'` · Chat prompt: "What is the largest planet in the solar system?"

**`base`** (60 tokens, hit limit, repetition 0.60)

````text
the Earth. It is the largest planet in the solar system.
The largest planet in the solar system is the Earth. It is the largest planet in the solar system.
The largest planet in the solar system is the Moon. It is the largest planet in the solar system.
The largest
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.80)

````text
The largest planet in the solar system is the Sun, which is the smallest planet in the solar system. It is located in the constellation of Cancer, which is the southern hemisphere of the solar system.

The largest planet in the solar system is the Earth, which is the smallest planet in the solar system. It is located in the constellation of Cancer, which is the southern hemisphere of the solar sys [...]
````

**`sft-50k`** (47 tokens, stopped, repetition 0.44)

````text
The largest planet in the solar system is Jupiter, which is approximately 4.5 times larger than Earth. The largest planet in the solar system is Jupiter, which is approximately 4.5 times larger than Earth.
````

### general-03 (general_knowledge)

Completion prompt: `'Water is made of hydrogen and'` · Chat prompt: "What is water made of?"

**`base`** (60 tokens, hit limit, repetition 0.77)

````text
oxygen.
- The water is heated to boiling point.
- The water is heated to boiling point.
- The water is heated to boiling point.
- The water is heated to boiling point.
- The water is heated to boiling
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.28)

````text
Water is made of water, which is a mixture of water and carbon dioxide. The water is heated to boiling point (100 degrees Celsius) and then condensed to form a gas called water vapor. The vapor is then condensed into a liquid called water vapor, which is then heated to boiling point (100 degrees Celsius).

Water is a fundamental element in the Earth's history, and its composition is a key factor i [...]
````

**`sft-50k`** (256 tokens, hit limit, repetition 0.28)

````text
Water is a fundamental element in the Earth's crust, and it plays a crucial role in the formation of rocks, minerals, and other materials. The primary water bodies include the oceans, lakes, rivers, and oceans, as well as the atmosphere, land, and oceans.

Water is a vital component of the Earth's climate system, as it plays a crucial role in regulating the Earth's climate. The oceans, which cover [...]
````

### history-01 (general_knowledge)

Completion prompt: `'The history of the Roman Empire began'` · Chat prompt: "How did the Roman Empire begin?"

**`base`** (60 tokens, hit limit, repetition 0.16)

````text
in the year 49 BCE. The Roman Empire was a vast empire that stretched from the Mediterranean to the Black Sea. The Roman Empire was a powerful and powerful empire that flourished from the year 49 BCE to the year 39 BCE. The Roman
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.81)

````text
The Roman Empire began in the year 56 BCE, and it was a time of great change and growth. It was a time of great change and growth, marked by great social and economic growth. The Roman Empire was a time of great change and growth, marked by great social and economic growth.

The Roman Empire was a time of great change and growth, marked by great social and economic growth. The Roman Empire was a t [...]
````

**`sft-50k`** (256 tokens, hit limit, repetition 0.45)

````text
The Roman Empire began in the year 4 BCE, during the reign of Emperor Nero, during the reign of Emperor Augustus. The Empire was a grand empire of Rome, with a capital at Rome. The Roman Empire was a great power, with a strong military, economic, and cultural foundation.

The Roman Empire was founded by the Roman Emperor Augustus, who was a member of the Council of Trent, a council of Roman govern [...]
````

### reasoning-01 (reasoning)

Completion prompt: `'If a train leaves the station at 3pm and travels for 2 hours, it arrives at'` · Chat prompt: "A train leaves the station at 3pm and travels for 2 hours. What time does it arrive?"

**`base`** (60 tokens, hit limit, repetition 0.65)

````text
the station at 4pm.
The train leaves the station at 4pm and travels for 2 hours.
The train leaves the station at 4pm and travels for 2 hours.
The train leaves the station at 4pm and travels for 2 hours.
The
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.82)

````text
The train leaves at 3pm and travels for 2 hours. The station is at 3pm and the train is at 4pm. The train is at 4pm and the station is at 4pm.

The train leaves the station at 4pm and travels for 2 hours. The station is at 4pm and the train is at 4pm.

The train leaves the station at 4pm and travels for 2 hours. The station is at 4pm and the train is at 4pm.

The train leaves the station at 4pm an [...]
````

**`sft-50k`** (122 tokens, stopped, repetition 0.27)

````text
To determine when the train leaves the station, we need to calculate the time it takes for the train to reach the station. Since the station is at 3pm, the train will be at 3pm.

The train starts at 3pm and leaves at 3pm. Since the station is at 3pm, the train will be at 3pm.

The time it takes for the train to reach the station is the following:
time = 3pm - 3pm = 3 hours

So, the train arrives a [...]
````

### reasoning-02 (reasoning)

Completion prompt: `'Two plus two equals'` · Chat prompt: "What is two plus two?"

**`base`** (60 tokens, hit limit, repetition 0.47)

````text
one.
Then, the second equals one plus two.
Then, the third equals one plus two.
Then, the fourth equals one plus two.
Then, the fifth equals one plus two.
Then, the sixth equals
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.92)

````text
The two plus two is a combination of two or more numbers. The two plus two is a combination of two or more numbers.

The two plus two is a combination of two or more numbers.

The two plus two is a combination of two or more numbers.

The two plus two is a combination of two or more numbers.

The two plus two is a combination of two or more numbers.

The two plus two is a combination of two or mor [...]
````

**`sft-50k`** (256 tokens, hit limit, repetition 0.87)

````text
In mathematics, two plus two is a fundamental concept that can be expressed as two plus two, two plus two, and two plus two. It's a fundamental concept that can be expressed as two plus two, two plus two, and two plus two, and it's a fundamental concept that can be expressed as two plus two, two plus two, and two plus two, and it's a fundamental concept that can be expressed as two plus two, two p [...]
````

### reasoning-03 (reasoning)

Completion prompt: `'If all cats are animals, and Whiskers is a cat, then Whiskers is'` · Chat prompt: "All cats are animals, and Whiskers is a cat. What can we conclude about Whiskers?"

**`base`** (60 tokens, hit limit, repetition 0.54)

````text
a cat.
Whiskers is a cat’s most common cat’s name. It is a cat’s most common name.
Whiskers is a cat’s most common name. It is a cat’s most common cat’s name.
Whiskers is
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.60)

````text
To determine Whiskers, we can start by examining the characteristics of the Whiskers. Whiskers are small, rounded ears that are often found in cats. They are often found in pairs or pairs, and they are often associated with territoriality and territoriality.


Whiskers are also known for their distinctive tail fin, which is a distinctive feature that distinguishes them from other cats. They are of [...]
````

**`sft-50k`** (256 tokens, hit limit, repetition 0.30)

````text
Whiskers is a fascinating cat species that has been studied extensively in various parts of the world. The study of Whiskers has led to several insights into the behavior, physiology, and behavior of cats.

One of the most significant discoveries is that Whiskers is a social cat, meaning it has a social structure. This social structure is often described as a "cat-like" structure, where the domina [...]
````

### code-01 (code)

Completion prompt: `'def fibonacci(n):'` · Chat prompt: "Write a Python function that returns the n-th Fibonacci number."

**`base`** (60 tokens, hit limit, repetition 0.82)

````text
- Fibonacci(n):
- Fibonacci(n):
- Fibonacci(n):
- Fibonacci(n):
- Fibonacci(n):
- Fibonacci(n):
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.83)

````text
Here's how you can implement this function in Python:

```python
def n-th_fibonacci_number(n):
    return n-th_fibonacci_number(n):
````

**`sft-50k`** (256 tokens, hit limit, repetition 0.42)

````text
Here's a Python function that uses the Fibonacci sequence to calculate the n-th Fibonacci number:

```python
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    elif n == 3:
        return 3
    elif n == 4:
        return 4
    elif n == 5:
        return 5
    elif n == 6:
        return 6
    elif n == 7:
        return 7
    [...]
````

### code-02 (code)

Completion prompt: `'def is_even(n):\n    return'` · Chat prompt: "Write a Python function that checks whether a number is even."

**`base`** (60 tokens, hit limit, repetition 0.93)

````text
s the value of the value of the value of the value of the value of the value of the value of the value of the value of the value of the value of the value of the value of the value of the value of the value of the value of the value of the value of the value
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.83)

````text
Here's how you can implement this:

```python
def find_finished(num):
    return num
    if num is not even:
````

**`sft-50k`** (256 tokens, hit limit, repetition 0.25)

````text
Here's a simple Python function that checks whether a number is even:

```python
def is_even(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True
```

This function works by checking if the input number `num` is even. If it is, it means that the number is even, so it returns `False`. Otherwise, it c [...]
````

### code-03 (code)

Completion prompt: `'# Function to reverse a string\ndef reverse_string(s):'` · Chat prompt: "Write a Python function that reverses a string."

**`base`** (60 tokens, hit limit, repetition 0.91)

````text
return a string that is a string that is a string that is a string that is a string that is a string that is a string that is a string that is a string that is a string that is a string that is a string that is a string that is a string that is a string that
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.81)

````text
Here's how you can implement it:

```python
def reverses_string(string):
    return reversed_string(string)
    return reversed_string(string)
```

Here's how you can implement it:

```python
def reverses_string(string):
    return reversed_string(string)
    return reversed_string(string)
```

Here's how you can implement it:

```python
def reverses_string(string):
    return reversed_string(stri [...]
````

**`sft-50k`** (97 tokens, stopped, repetition 0.13)

````text
Here's a simple Python function that reverses a string:

```python
def reverse_string(s):
    return s[::-1]
```

You can use this function like this:

```python
print(reverse_string("hello"))  # Output: "hello"
```
````

### writing-01 (writing)

Completion prompt: `'Once upon a time, in a small village,'` · Chat prompt: "Write the opening of a short story set in a small village."

**`base`** (60 tokens, hit limit, repetition 0.60)

````text
the village was a place of refuge. It was a place of refuge for the poor and the sick.
The village was a place of refuge for the poor and the sick. It was a place of refuge for the poor and the sick. It was a place of refuge for the poor and the
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.98)

````text
The village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the village of the villag [...]
````

**`sft-50k`** (256 tokens, hit limit, repetition 0.13)

````text
In the village of Ravenswood, a young woman named Rachel lived in a small village, surrounded by the rolling hills and the forest. She was a hardworking and independent woman, but she had always felt a deep connection to nature and the land around her. She had always been drawn to the natural world, and she knew that the village would be a place where she could find solace and inspiration.

As she [...]
````

### writing-02 (writing)

Completion prompt: `'The weather today is'` · Chat prompt: "Write a few sentences describing the weather today."

**`base`** (60 tokens, hit limit, repetition 0.74)

````text
a bit different from the weather in the past.
The weather today is a bit different from the weather in the past.
The weather today is a bit different from the weather in the past.
The weather today is a bit different from the weather in the past.
The weather today is
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.75)

````text
The weather today is a bit colder than it was in the 1930s. The average temperature in the region is about 2 degrees Celsius (3.7 degrees Fahrenheit). The average daily temperature is about 1.5 degrees Celsius (3.7 degrees Fahrenheit).

The average daily temperature in the region is about 2 degrees Celsius (3.7 degrees Fahrenheit). The average daily daily temperature in the region is about 1.5 deg [...]
````

**`sft-50k`** (77 tokens, stopped, repetition 0.03)

````text
Today's forecast shows a mix of sunny skies and mild weather. The sunny skies are expected to remain strong this afternoon, with temperatures ranging from 70°F to 80°F (22°C to 23°C). The mild weather will bring a mix of sunshine and rain, making it a perfect day for outdoor activities.
````

### writing-03 (writing)

Completion prompt: `'My favorite hobby is'` · Chat prompt: "Suggest a hobby I could start, and explain why it's enjoyable."

**`base`** (60 tokens, hit limit, repetition 0.77)

````text
to make a hobby of your own.
- Make a hobby of your own.
- Make a hobby of your own.
- Make a hobby of your own.
- Make a hobby of your own.
- Make a hob
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.49)

````text
I'd like to share some of the reasons why it's enjoyable.

Firstly, it's a great way to get out of the comfort zone and get out of the comfort zone. This can be achieved by taking a walk, taking a stroll, or simply taking a few minutes to relax.

Next, it can be a great way to get out of the comfort zone and get out of the comfort zone. This can be achieved by taking a few minutes to walk, take a  [...]
````

**`sft-50k`** (256 tokens, hit limit, repetition 0.42)

````text
Starting a hobby can be a great way to break the bank account and get some exercise. Here's a suggestion: Instead of starting with a simple hobby, you could start with a more complex activity like a game of cards, a game of golf, or even a game of fetch. This will give you a sense of excitement and can help you stay motivated.

You could also try a new hobby, like a new hobby, like a new painting, [...]
````

### research-01 (research)

Completion prompt: `'A transformer model works by'` · Chat prompt: "How does a transformer model work?"

**`base`** (60 tokens, hit limit, repetition 0.61)

````text
converting the voltage to the current.
The transformer is a device that converts the voltage to the current. The transformer is a device that converts the voltage to the current. The transformer is a device that converts the voltage to the current. The transformer is a device
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.61)

````text
To design a transformer model, we need to consider the transformer's characteristics, such as its size, power, and operating temperature. The transformer's primary purpose is to convert the direct current (DC) energy from the primary power supply to the alternating current (AC) energy from the secondary power supply.

To achieve this, we need to consider the transformer's voltage and current requi [...]
````

**`sft-50k`** (256 tokens, hit limit, repetition 0.48)

````text
A transformer model is a type of electrical engineering device that models the behavior of a transformer in a specific application. It's a device that uses a combination of electrical and mechanical principles to transform a transformer into a stable, reliable, and efficient operation.

The transformer model typically consists of a series of interconnected transformers, which are connected to a se [...]
````

### research-02 (research)

Completion prompt: `'The difference between supervised and unsupervised learning is'` · Chat prompt: "What is the difference between supervised and unsupervised learning?"

**`base`** (60 tokens, hit limit, repetition 0.74)

````text
that supervised learning is the learning of a set of skills, which is the learning of a set of skills, which is the learning of a set of skills, which is the learning of a set of skills, which is the learning of a set of skills, which is the learning of a
````

**`sft-2k`** (256 tokens, hit limit, repetition 0.75)

````text
The difference between supervised and unsupervised learning is that supervised learning is where the learner is given a specific task or task and the learner is given a specific set of tasks or tasks to complete. The learner is given a specific set of tasks and the learner is given a specific set of tasks to complete.

In supervised learning, the learner is given a specific set of tasks and the le [...]
````

**`sft-50k`** (256 tokens, hit limit, repetition 0.26)

````text
In data analysis, supervised and unsupervised learning are two distinct approaches to data analysis.

**Similar to supervised learning, where the data is manipulated and analyzed using algorithms, supervised learning is used to train a model that can handle unstructured data. It involves using a model to predict outcomes, rather than relying on the data itself.

**Unsupervised learning, on the oth [...]
````
