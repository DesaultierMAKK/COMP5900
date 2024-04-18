import os
import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from tqdm import trange

from buffer import Buffer
from mutation import Mutations
from tournament import TournamentSelection
from utils import initialPopulation

class MultiAgentTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize configurations
        self.initialize_configurations()
        
        # Initialize the environment
        self.env = self.initialize_environment()
        
        # Initialize the multi-agent replay buffer
        self.memory = self.initialize_replay_buffer()
        
        # Initialize the population
        self.pop = self.initialize_population()
        
        # Initialize mutation and selection strategies
        self.tournament = self.initialize_tournament_selection()
        self.mutations = self.initialize_mutations()
        
        # Training loop parameters
        self.max_episodes = 500
        self.max_steps = 25
        self.epsilon = 1.0
        self.eps_end = 0.1
        self.eps_decay = 0.995
        self.evo_epochs = 20
        self.evo_loop = 1
        self.elite = self.pop[0]
        
    def initialize_configurations(self):
        """Initialize configurations for the training."""
        self.net_config = {
            "arch": "mlp",
            "hidden_size": [32, 32],
        }
        
        self.init_hp = {
            "POPULATION_SIZE": 4,
            "ALGO": "MATD3",
            "CHANNELS_LAST": False,
            "BATCH_SIZE": 32,
            "LR_ACTOR": 0.001,
            "LR_CRITIC": 0.01,
            "GAMMA": 0.95,
            "MEMORY_SIZE": 100000,
            "LEARN_STEP": 5,
            "TAU": 0.01,
            "POLICY_FREQ": 2,
        }
        
    def initialize_environment(self):
        """Initialize the simple speaker listener environment as a parallel environment."""
        env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
        env.reset()
        
        # Configure the multi-agent algo input arguments
        try:
            state_dim = [env.observation_space(agent).n for agent in env.agents]
            self.one_hot = True
        except Exception:
            state_dim = [env.observation_space(agent).shape for agent in env.agents]
            self.one_hot = False
        
        try:
            action_dim = [env.action_space(agent).n for agent in env.agents]
            self.init_hp["DISCRETE_ACTIONS"] = True
            self.init_hp["MAX_ACTION"] = None
            self.init_hp["MIN_ACTION"] = None
        except Exception:
            action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
            self.init_hp["DISCRETE_ACTIONS"] = False
            self.init_hp["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
            self.init_hp["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]
        
        if self.init_hp["CHANNELS_LAST"]:
            state_dim = [
                (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
            ]
        
        # Append number of agents and agent IDs to the initial hyperparameter dictionary
        self.init_hp["N_AGENTS"] = env.num_agents
        self.init_hp["AGENT_IDS"] = env.agents
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        return env
    
    def initialize_replay_buffer(self):
        """Initialize the multi-agent replay buffer."""
        field_names = ["state", "action", "reward", "next_state", "done"]
        memory = Buffer(
            self.init_hp["MEMORY_SIZE"],
            field_names=field_names,
            agent_ids=self.init_hp["AGENT_IDS"],
            device=self.device,
        )
        return memory
    
    def initialize_population(self):
        """Create a population ready for evolutionary hyperparameter optimization."""
        pop = initialPopulation(
            self.init_hp["ALGO"],
            self.state_dim,
            self.action_dim,
            self.one_hot,
            self.net_config,
            self.init_hp,
            population_size=self.init_hp["POPULATION_SIZE"],
            device=self.device,
        )
        return pop
    
    def initialize_tournament_selection(self):
        """Instantiate a tournament selection object (used for HPO)."""
        tournament = TournamentSelection(
            tournament_size=2,
            elitism=True,
            population_size=self.init_hp["POPULATION_SIZE"],
            evo_step=1,
        )
        return tournament
    
    def initialize_mutations(self):
        """Instantiate a mutations object (used for HPO)."""
        mutations = Mutations(
            algo=self.init_hp["ALGO"],
            no_mutation=0.2,
            architecture=0.2,
            new_layer_prob=0.2,
            parameters=0.2,
            activation=0,
            rl_hp=0.2,
            rl_hp_selection=[
                "lr",
                "learn_step",
                "batch_size",
            ],
            mutation_sd=0.1,
            agent_ids=self.init_hp["AGENT_IDS"],
            arch=self.net_config["arch"],
            rand_seed=1,
            device=self.device,
        )
        return mutations
    
    def train(self):
        """Main training loop."""
        for idx_epi in trange(self.max_episodes):
            for agent in self.pop:  # Loop through population
                state, info = self.env.reset()  # Reset environment at start of episode
                agent_reward = {agent_id: 0 for agent_id in self.env.agents}
                
                if self.init_hp["CHANNELS_LAST"]:
                    state = {
                        agent_id: np.moveaxis(np.expand_dims(s, 0), [-1], [-3])
                        for agent_id, s in state.items()
                    }
                
                # Run an episode
                for _ in range(self.max_steps):
                    agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
                    env_defined_actions = (
                        info["env_defined_actions"]
                        if "env_defined_actions" in info.keys()
                        else None
                    )
                    
                    # Get next action from agent
                    cont_actions, discrete_action = agent.getAction(
                        state, self.epsilon, agent_mask, env_defined_actions
                    )
                    
                    if agent.discrete_actions:
                        action = discrete_action
                    else:
                        action = cont_actions
                    
                    next_state, reward, termination, truncation, info = self.env.step(
                        action
                    )
                    
                    # Process state and next state if channels_last is set
                    state, next_state = self.process_states(state, next_state)
                    
                    # Save experiences to replay buffer
                    self.memory.save2memory(state, cont_actions, reward, next_state, termination)
                    
                    # Collect the reward
                    self.collect_reward(agent_reward, reward)
                    
                    # Learn according to learning frequency
                    if self.should_learn(agent):
                        experiences = self.memory.sample(agent.batch_size)
                        agent.learn(experiences)
                    
                    # Update the state
                    state = next_state
                    
                    # Stop episode if any agents have terminated
                    if any(truncation.values()) or any(termination.values()):
                        break
                
                # Save the total episode reward
                score = sum(agent_reward.values())
                agent.scores.append(score)
            
            # Update epsilon for exploration
            self.update_epsilon()
            
            # Evolve population if necessary
            if (idx_epi + 1) % self.evo_epochs == 0:
                self.evolve_population(idx_epi)
        
        # Save the trained algorithm
        self.save_trained_algorithm()
    
    def process_states(self, state, next_state):
        """Process state and next state based on the CHANNELS_LAST setting."""
        if self.init_hp["CHANNELS_LAST"]:
            state = {agent_id: np.squeeze(s) for agent_id, s in state.items()}
            next_state = {
                agent_id: np.moveaxis(ns, [-1], [-3])
                for agent_id, ns in next_state.items()
            }
        return state, next_state
    
    def collect_reward(self, agent_reward, reward):
        """Collect the reward for each agent."""
        for agent_id, r in reward.items():
            agent_reward[agent_id] += r
    
    def should_learn(self, agent):
        """Check if the agent should learn based on learning frequency and memory."""
        return (
            (self.memory.counter % agent.learn_step == 0) and
            (len(self.memory) >= agent.batch_size)
        )
    
    def update_epsilon(self):
        """Update epsilon for exploration."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
    
    def evolve_population(self, idx_epi):
        """Evaluate and evolve the population."""
        # Evaluate population
        fitnesses = [
            agent.test(
                self.env,
                swap_channels=self.init_hp["CHANNELS_LAST"],
                max_steps=self.max_steps,
                loop=self.evo_loop,
            )
            for agent in self.pop
        ]
        
        print(f"Episode {idx_epi + 1}/{self.max_episodes}")
        print(f'Fitnesses: {["%.2f" % fitness for fitness in fitnesses]}')
        print(
            f'100 fitness avgs: {["%.2f" % np.mean(agent.fitness[-100:]) for agent in self.pop]}'
        )
        
        # Tournament selection and population mutation
        self.elite, self.pop = self.tournament.select(self.pop)
        self.pop = self.mutations.mutation(self.pop)
    
    def save_trained_algorithm(self):
        """Save the trained algorithm."""
        path = "./"
        filename = "trained_matd3.pt"
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        self.elite.saveCheckpoint(save_path)

if __name__ == "__main__":
    trainer = MultiAgentTrainer()
    trainer.train()
