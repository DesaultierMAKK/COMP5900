import os
import imageio
import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from PIL import Image, ImageDraw

from matd3 import MATD3


# Define function to return image
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(frame) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] / 20, im.size[1] / 18), f"Episode: {episode_num + 1}", fill=text_color
    )

    return im


class MultiAgentTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure the environment
        self.env = simple_speaker_listener_v4.parallel_env(
            continuous_actions=True, render_mode="rgb_array"
        )
        self.env.reset()
        try:
            state_dim = [self.env.observation_space(agent).n for agent in self.env.agents]
            one_hot = True
        except Exception:
            state_dim = [self.env.observation_space(agent).shape for agent in self.env.agents]
            one_hot = False
        try:
            action_dim = [self.env.action_space(agent).n for agent in self.env.agents]
            discrete_actions = True
            max_action = None
            min_action = None
        except Exception:
            action_dim = [self.env.action_space(agent).shape[0] for agent in self.env.agents]
            discrete_actions = False
            max_action = [self.env.action_space(agent).high for agent in self.env.agents]
            min_action = [self.env.action_space(agent).low for agent in self.env.agents]

        # Append number of agents and agent IDs to the initial hyperparameter dictionary
        n_agents = self.env.num_agents
        agent_ids = self.env.agents

        # Instantiate an MATD3 object
        self.matd3 = MATD3(
            state_dim,
            action_dim,
            one_hot,
            n_agents,
            agent_ids,
            max_action,
            min_action,
            discrete_actions,
            device=self.device,
        )

        # Load the saved algorithm into the MATD3 object
        path = "./trained_matd3.pt"
        self.matd3.loadCheckpoint(path)

        # Define test loop parameters
        self.episodes = 10  # Number of episodes to test agent on
        self.max_steps = 25  # Max number of steps to take in the environment in each episode

        self.rewards = []  # List to collect total episodic reward
        self.frames = []  # List to collect frames
        self.indi_agent_rewards = {
            agent_id: [] for agent_id in agent_ids
        }  # Dictionary to collect individual agent rewards

        self.gif_path = "./"

    def run(self):
        """Run the testing and GIF saving process."""
        self.run_test_loop()
        self.save_gif()
        print(f"GIF saved at: {os.path.join(self.gif_path, 'evaluated_matd3.gif')}")

    def run_test_loop(self):
        """Run the testing loop for inference."""
        for ep in range(self.episodes):
            state, info = self.env.reset()
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            score = 0
            for _ in range(self.max_steps):
                agent_mask = info["agent_mask"] if "agent_mask" in info else None
                env_defined_actions = info.get("env_defined_actions", None)

                # Get next action from agent
                cont_actions, discrete_action = self.matd3.getAction(
                    state,
                    epsilon=0,
                    agent_mask=agent_mask,
                    env_defined_actions=env_defined_actions,
                )
                if self.matd3.discrete_actions:
                    action = discrete_action
                else:
                    action = cont_actions

                # Save the frame for this step and append to frames list
                frame = self.env.render()
                self.frames.append(_label_with_episode_number(frame, episode_num=ep))

                # Take action in environment
                state, reward, termination, truncation, info = self.env.step(action)

                # Save agent's reward for this step in this episode
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                # Determine total score for the episode and then append to rewards list
                score = sum(agent_reward.values())

                # Stop episode if any agents have terminated
                if any(truncation.values()) or any(termination.values()):
                    break

            self.rewards.append(score)

            # Record agent-specific episodic reward
            for agent_id in self.env.agents:
                self.indi_agent_rewards[agent_id].append(agent_reward[agent_id])

            self.print_episode_results(ep)

        # Close the environment when testing is done
        self.env.close()

    def print_episode_results(self, ep):
        """Print the results for an episode."""
        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print("Episodic Reward: ", self.rewards[-1])
        for agent_id, reward_list in self.indi_agent_rewards.items():
            # Check if the reward list is not empty
            if reward_list:
                print(f"{agent_id} reward: {reward_list[-1]}")
            else:
                print(f"{agent_id} reward: No rewards recorded for this agent")

    def save_gif(self):
        """Save the collected frames as a GIF."""
        os.makedirs(self.gif_path, exist_ok=True)
        imageio.mimwrite(
            os.path.join(self.gif_path, "evaluated_matd3.gif"), self.frames, duration=10
        )


if __name__ == "__main__":
    tester = MultiAgentTester()
    tester.run()
