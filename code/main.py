from transformers import pipeline

classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

text = "I'm feeling extremely anxious and can't sleep."
result = classifier(text)
print(result)

# State management for dialogue systems
class DialogueStateTracker:
    def __init__(self):
        self.state = {
            "slots": {},
            "intents": [],
            "emotions": [],
            "history": []
        }

    def update(self, user_input, intent, slots, emotion):
        self.state["history"].append(user_input)
        self.state["intents"].append(intent)
        self.state["emotions"].append(emotion)
        for slot, value in slots.items():
            self.state["slots"][slot] = value

    def get_state(self):
        return self.state

