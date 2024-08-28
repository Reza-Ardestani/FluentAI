## Report
Disfluency Correction in Questions 

<strong>⚠️ In this part I will report some of the details of development.</strong>

## Table of Contents

- [Preliminary Research](#preliminary-research)
- [Sprint 1](#sprint-1)
    - [Thematically analyzed 100 records](#thematically-analyzed-100-records)
    - [Filled Missing Values](#filled-missing-values)
    - [Early Stopping](#early-stopping)
    - [Exploratory Data Analyses](#exploratory-data-analyses)
- [Sprint 2](#sprint-2)
- [Results Summary](#results-summary)
- [Future Works](#future-works)
- [References](#references)


## Preliminary Research
.

## Development Summary

## Sprint 1

### Thematically analyzed 100 records
Please refer to the results folder for downloading the 100 records file with the least GLEU score. I thematically analyzed 100 records and here are the types:

1. Generated questions are better than the reference question: 3421, 460, 1689, 3608,
2. Missing value for the input ("disfluent": "#VALUE!”) in 13 cases like document 166
3. Need for a better metric that captures semantic similarity: 2287, 931, 795, 2264, 2301
4. Coreference disambiguating needed: 410, 3117, 2289, 940
5. Wrong generated dis-fluent question: 144, 117, 675

### Filled Missing Values

I used GPT4-o and prompt engineering to fill the missing values of train and development datasets. The prompt is as follows:


~~~
Task: Disfluency Generation in Questions

You are an intelligent AI assistant tasked with generating disfluent versions of fluent questions. A disfluent question contains errors, hesitations, or corrections that resemble natural speech patterns. There are three main types of disfluency to consider:
1. Repetition: A part of the question is repeated with a hesitation or error (e.g., “When is Eas ugh Easter this year?”).
2. Correction: A word or phrase is replaced after a momentary mistake (e.g., “When is Lent I meant Easter this year?”).
3. Restarts: The speaker begins the question with one phrase but then stops and starts over with the correct phrasing (e.g., “How much no wait when is Easter this year?”).

Components of a Disfluent Question:
* Reparandum: The part of the question that is intended to be replaced or ignored.
* Interregnum: Optional words or phrases that signal a correction or pause (e.g., "no wait", "I mean", "ugh”, “scratch that”, “ahm” or other correction words or signals).
* Repair: The corrected word or phrase that replaces the reparandum.

Example:
* Original Question: "When is Easter this year?"
* Disfluent Versions:
    * Repetition: "When is Eas ugh Easter this year?"
    * Correction: "When is Lent I meant Easter this year?"
    * Restarts: "How much no wait when is Easter this year?"

More example:
1. Original: "In what country is Normandy located?"
* Disfluent: "In what country is Norse found no wait Normandy not Norse?"
2. Original: "When were the Normans in Normandy?"
* Disfluent: "From which countries no tell me when were the Normans in Normandy?"
3. Original: "From which countries did the Norse originate?"
* Disfluent: "From which Norse leader I mean countries did the Norse originate?"
4. Original: "Who was the Norse leader?"
* Disfluent: "When I mean Who was the Norse leader?"
5. Original: "What century did the Normans first gain their separate identity?"
* Disfluent: "When no what century did the Normans first gain their separate identity?"

Your Task:
Using one of the three categories of disfluency (Repetition, Correction, or Restarts), be creative and generate a only one disfluent version of the following original question:
Original Question: [ENETER THE QUESTION HERE]
~~~


### Early Stopping
![Train and Validation Loss v2.1](imgs/train_validation_loss_v2.1.png)

### Exploratory Data Analyses
 have conducted comprehensive Exploratory Data Analyses (EDA) to gain a deeper understanding of the dataset and uncover insights crucial for model optimization. Through this process, I was able to identify key characteristics of the data, which informed the selection of optimal hyperparameters such as the maximum input length, ultimately enhancing the model’s performance and efficiency.
 The following picture shows the token lenght distribution in the dataset base on which I set the value for "max_input_lenght" hyper parameter.
 
![max input lenght](imgs/Unknown-3.png)

## Sprint 2


## Results Summary


## Future Works



## References

The complete list of references are in the report file.
<br><br>
[1] DISFL-QA: A Benchmark Dataset for Understanding Disfluencies in Question Answering

[2] My toturial: [https://youtu.be/wIkIxrsnUs8?si=3NtkXKgC87_Y7w2H&t=1515](https://youtu.be/wIkIxrsnUs8?si=3NtkXKgC87_Y7w2H&t=1515)



<br>
<br>
<br>

### Thank you for your interest in my AI development


<br>
<br>
<br>

