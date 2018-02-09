# Switcher demo

Module that analyses messages from user and determines when to switch chatbot to live operator

Using: `Python 3.6`, `TensorFlow`, `Keras`

## Installation

- clone the project
`git clone [url_to_project]`
- install requirements:
`pip3 install -r requirements.txt`

## Using

Usage is simple:

```bash
python3 switcher.py "[text of message]"
```

Here is `[text of message]` is the text of user message.
Script returns a probability that current user phrase must call an operator.


Examples:
```bash
$ python switcher.py "Cucumbers is cool"
Probability that operator is needed - 0%
```
```bash
$ python switcher.py "Call the operator"
Probability that operator is needed - 99%
```
```bash
$ python switcher.py "You stupid bot"
Probability that operator is needed - 4%
```
```bash
$ python switcher.py "Need a real person"
Probability that operator is needed - 30%
```
```bash
$ python switcher.py "What is your name?"
Probability that operator is needed - 0%
```