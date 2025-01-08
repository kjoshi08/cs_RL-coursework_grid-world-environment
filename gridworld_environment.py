import numpy as np
import matplotlib.pyplot as plt


# Mock MDP class and build_mazeMDP function
class MockMDP:
    def __init__(self):
        self.T = np.zeros((4, 16, 16))
        for s in range(16):
            for a in range(4):
                next_s = (s + a + 1) % 16  # Example: deterministic transitions
                self.T[a, s, next_s] = 1.0
        self.R = np.random.uniform(-1, 1, (4, 16))  # Random rewards
        self.discount = 0.9


def build_mazeMDP():
    return MockMDP()


# ReinforcementLearning class
class ReinforcementLearning:
    def __init__(self, mdp):
        self.mdp = mdp


    def sampleRewardAndNextState(self, state, action):
        prob = self.mdp.T[action, state]
        next_state = np.random.choice(len(prob), p=prob)
        reward = self.mdp.R[action, state]
        return reward, next_state


    def reinforce(self, theta=None, alpha=0.01, n_episodes=3000):
        if theta is None:
            theta = np.random.rand(self.mdp.R.shape[0], self.mdp.R.shape[1])
        cum_rewards = []


        for _ in range(n_episodes):
            state = 0
            episode = []
            rewards = []


            while state != 15:
                probs = np.exp(theta[:, state]) / np.sum(np.exp(theta[:, state]))
                action = np.random.choice(len(probs), p=probs)
                reward, next_state = self.sampleRewardAndNextState(state, action)
                episode.append((state, action))
                rewards.append(reward)
                state = next_state


            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + self.mdp.discount * G
                returns.insert(0, G)


            for t, (state, action) in enumerate(episode):
                probs = np.exp(theta[:, state]) / np.sum(np.exp(theta[:, state]))
                grad_log = np.zeros_like(theta)
                grad_log[action, state] = 1 - probs[action]
                theta += alpha * grad_log * returns[t]


            cum_rewards.append(sum(rewards))


        return cum_rewards, theta


    def actorCritic(self, theta=None, alpha_theta=0.01, alpha_w=0.01, n_episodes=3000):
        if theta is None:
            theta = np.random.rand(self.mdp.R.shape[0], self.mdp.R.shape[1])
        w = np.zeros(self.mdp.R.shape[1])
        cum_rewards = []


        for _ in range(n_episodes):
            state = 0
            rewards = []


            while state != 15:
                probs = np.exp(theta[:, state]) / np.sum(np.exp(theta[:, state]))
                action = np.random.choice(len(probs), p=probs)
                reward, next_state = self.sampleRewardAndNextState(state, action)


                td_error = reward + self.mdp.discount * w[next_state] - w[state]
                w[state] += alpha_w * td_error


                grad_log = np.zeros_like(theta)
                grad_log[action, state] = 1 - probs[action]
                theta += alpha_theta * grad_log * td_error


                rewards.append(reward)
                state = next_state


            cum_rewards.append(sum(rewards))


        return cum_rewards, theta, w


# Main execution
if __name__ == "__main__":
    mdp = build_mazeMDP()
    rl = ReinforcementLearning(mdp)


    n_episode = 100  # Reduced for debugging
    n_trials = 3


    # Evaluate REINFORCE
    out_reinforce = np.zeros([n_trials, n_episode])
    for i in range(n_trials):
        cum_rewards, _ = rl.reinforce(n_episodes=n_episode)
        out_reinforce[i, :] = np.array(cum_rewards)
        print(f"REINFORCE Trial {i + 1}: Cumulative rewards = {cum_rewards[:10]}")  # Debug output


    # Evaluate Actor-Critic
    out_actor_critic = np.zeros([n_trials, n_episode])
    for i in range(n_trials):
        cum_rewards, _, _ = rl.actorCritic(n_episodes=n_episode)
        out_actor_critic[i] = cum_rewards
        print(f"Actor-Critic Trial {i + 1}: Cumulative rewards = {cum_rewards[:10]}")  # Debug output


    # Plot results
    plt.plot(out_reinforce.mean(axis=0), label='REINFORCE')
    plt.plot(out_actor_critic.mean(axis=0), label='Actor-Critic')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Policy Gradient Algorithms Performance')
    plt.legend()
    plt.show()