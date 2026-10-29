# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Versioned adult utterances. Text describes the scene; only audio reaches MIMo."""
CONTRACT = 'responsive_exchange_v1'
GROUPS = {
    'greeting': ["Hello, little one. I'm here.", "Hello there. Let's spend some time together.",
                 "Hi. I'm right here with you.", "Hello. It's good to see you."],
    'return': ["Hello again. I'm back.", "Here I am again.", "I'm back. Hello there.",
               "Hello, little one. I'm here again."],
    'farewell': ["I'm going to step away. I'll come back.", "I'll be nearby. See you in a little while.",
                 "I'm putting these away, then stepping away for a little while.",
                 "I'll be back. Let's have a little break."],
    'quiet': ["I'm here. We can be quiet together.", "Let's listen for a little while.",
              "We can have a quiet moment together."],
    'face_look': ["Hello. I see you looking this way.", "Here I am. Hello!",
                  "You're looking toward me. Hi there.", "Hello, little one."],
    'put_away': ["I'll put this away for a little while.", "This one is going away for now.",
                 "Let's put this one away.", "Away it goes. We can see it again later."],
}
OBJECT_NAMES = ('ball', 'pad', 'ring')
for name in OBJECT_NAMES:
    description = {'ball':'round ball', 'pad':'soft pad', 'ring':'round ring'}[name]
    GROUPS['present_'+name] = [f"Here's the {name}.", f"Look, a {description}.",
        f"I've brought the {name} over.", f"Here comes the {name}.", f"A {name}. Here it is."]
    GROUPS['touch_'+name] = [f"You're touching the {name}.", f"The {name} is touching you.",
        f"There's the {name}, against you.", f"There's the {name}, touching you."]
    GROUPS['look_'+name] = [f"You're looking at the {name}.",
        f"That is the {name}.", f"The {name}. There it is."]
    GROUPS['move_'+name] = [f"The {name} is moving.", f"There goes the {name}.",
        f"See the {name} move."]

VARIANTS = {f'exchange_{group}_{i}': text for group,texts in GROUPS.items() for i,text in enumerate(texts)}
GROUP_KEYS = {group:[f'exchange_{group}_{i}' for i in range(len(texts))] for group,texts in GROUPS.items()}
