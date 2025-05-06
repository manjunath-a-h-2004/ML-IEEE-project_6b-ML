# pvt_main.py

import random
from pvt_models import QLearningTutor, Bandit

def run_sessions(num_sessions=20):
    """
    Simulate num_sessions of learning and return the list of scores.
    """
    topics = ["Variables", "Loops", "Functions", "Recursion"]
    tutor = QLearningTutor(actions=topics)
    bandit = Bandit(actions=["video", "quiz", "text", "simulation"])

    scores = []
    state = "start"

    for session in range(num_sessions):
        # Q-learning chooses the topic
        action_index = tutor.choose_action(state)
        selected_topic = topics[action_index]

        # Bandit chooses content format (not used further here)
        bandit_action = bandit.choose_action()

        # Simulate a score
        score = random.randint(0, 100)
        reward = 1 if score >= 60 else -1

        # Update both models
        next_state = selected_topic
        tutor.update(state, action_index, reward, next_state)
        bandit.update(bandit_action, reward)

        state = next_state
        scores.append(score)

    return scores

if __name__ == "__main__":
    # If run directly, print a quick test
    print("Simulated scores:", run_sessions())
