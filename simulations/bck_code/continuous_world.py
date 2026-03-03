import gymnasium as gym
import numpy as np
import torch
import sys
import os
import glob
import zipfile
import io
import warnings
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from stable_baselines3 import PPO, DQN

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ALGO = "PPO"            # Options: "PPO", "DQN"
REGIMEN = "curriculum"  # Options: "curriculum", "noncurriculum"
NUM_EPISODES = 3

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. MODEL LOADER (From watch_agent.py)
# ==============================================================================
def load_clean_sb3(algo_class, path, env):
    print(f"Loading {algo_class.__name__} from {path}...")
    try:
        params = None
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("policy.pth") as f:
                params = torch.load(io.BytesIO(f.read()), map_location="cpu")
        
        # Try Default Architecture first
        try:
            model = algo_class("MlpPolicy", env=env, device="cpu")
            model.policy.load_state_dict(params)
            print("Loaded successfully (Default Arch).")
            return model
        except RuntimeError:
            print("Default arch mismatch. Retrying with [256, 256]...")
            policy_kwargs = dict(net_arch=[256, 256])
            model = algo_class("MlpPolicy", env=env, policy_kwargs=policy_kwargs, device="cpu")
            model.policy.load_state_dict(params)
            print("Loaded successfully (Custom Arch [256, 256]).")
            return model

    except Exception as e:
        print(f"Load Failed: {e}")
        return None

# ==============================================================================
# 2. CONTINUOUS ENVIRONMENT
# ==============================================================================
class ContinuousComplexEnv(AbstractEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "duration": 500,
            "simulation_frequency": 15,
            "policy_frequency": 5, # Match training frequency (important!)
            "screen_width": 1000,
            "screen_height": 600,
            "centering_position": [0.3, 0.5],
            "scaling": 6,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False
        })
        return config

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _reward(self, action: int) -> float:
        return 0 # Not needed for visualization

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _make_road(self):
        net = RoadNetwork()
        
        # Constants
        highway_start_x = 0
        highway_length = 300
        merge_length = 100
        roundabout_center_x = highway_length + merge_length + 40
        roundabout_radius = 30

        # 1. HIGHWAY
        for i in range(3):
            net.add_lane("start", "merge_start", StraightLane(
                [highway_start_x, i * 4], [highway_length, i * 4], 
                line_types=(LineType.CONTINUOUS if i == 0 else LineType.STRIPED, LineType.STRIPED if i < 2 else LineType.CONTINUOUS)
            ))

        # 2. MERGE
        net.add_lane("merge_start", "roundabout_entry", SineLane(
            [highway_length, 4], [roundabout_center_x - roundabout_radius, 0],
            amplitude=2, pulsation=2*np.pi / (merge_length), phase=0,
            line_types=[LineType.CONTINUOUS, LineType.CONTINUOUS]
        ))

        # 3. ROUNDABOUT (CCW)
        net.add_lane("roundabout_entry", "roundabout_exit", CircularLane(
            center=[roundabout_center_x, 0], radius=roundabout_radius,
            start_phase=np.pi, end_phase=2*np.pi, clockwise=False,
            line_types=[LineType.CONTINUOUS, LineType.NONE]
        ))
        net.add_lane("roundabout_exit", "roundabout_entry", CircularLane(
            center=[roundabout_center_x, 0], radius=roundabout_radius,
            start_phase=0, end_phase=np.pi, clockwise=False,
            line_types=[LineType.CONTINUOUS, LineType.NONE]
        ))

        # 4. EXIT
        net.add_lane("roundabout_exit", "finish", StraightLane(
            [roundabout_center_x + roundabout_radius, 0],
            [roundabout_center_x + roundabout_radius + 200, 0],
            line_types=[LineType.CONTINUOUS, LineType.CONTINUOUS]
        ))

        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _make_vehicles(self):
        # Ego
        self.vehicle = self.action_type.vehicle_class(
            self.road,
            self.road.network.get_lane(("start", "merge_start", 1)).position(0, 0),
            speed=20,
            heading=self.road.network.get_lane(("start", "merge_start", 1)).heading_at(0)
        )
        self.road.vehicles.append(self.vehicle)

        # Traffic
        for i in range(3):
            self.road.vehicles.append(self.action_type.vehicle_class(
                self.road,
                self.road.network.get_lane(("start", "merge_start", i)).position(50 + i*30, 0),
                speed=18
            ))
        self.road.vehicles.append(self.action_type.vehicle_class(
            self.road,
            self.road.network.get_lane(("roundabout_exit", "roundabout_entry", 0)).position(10, 0),
            speed=10
        ))

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    if "Continuous-v0" in gym.envs.registry:
        del gym.envs.registry["Continuous-v0"]
    gym.register(id='Continuous-v0', entry_point=ContinuousComplexEnv)

    # 1. Locate Model
    base_path = f"experiments/{ALGO}/{REGIMEN}/seed_0"
    run_dirs = glob.glob(os.path.join(base_path, "run_*"))
    if not run_dirs:
        print(f"No run found for {ALGO}/{REGIMEN}")
        sys.exit()
    latest_run = max(run_dirs, key=os.path.getmtime)
    model_path = os.path.join(latest_run, "final_model.zip")

    # 2. Wrapper (Required for 7-action compatibility)
    class PreferredLaneOvertakeWrapper(gym.Wrapper):
        def __init__(self, env, preferred_lane=1, speed_limit=33.0):
            super().__init__(env)
            self.preferred_lane = preferred_lane
            self.macro_actions = [
                {"base": 0}, {"base": 1}, {"base": 2}, {"base": 3}, 
                {"base": 4}, {"base": 4}, {"kind": "return"}
            ]
            self.action_space = gym.spaces.Discrete(len(self.macro_actions))
        def step(self, action):
            macro_idx = int(action) % len(self.macro_actions)
            macro = self.macro_actions[macro_idx]
            base_action = macro.get("base", 0)
            if macro.get("kind") == "return":
                base_action = self._return_to_preferred_lane()
            return self.env.step(base_action)
            
        def _return_to_preferred_lane(self):
            vehicle = getattr(self.env.unwrapped, "vehicle", None)
            if not vehicle: return 0
            lane = vehicle.lane_index[2] if len(vehicle.lane_index) >=3 else vehicle.lane_index
            if lane > self.preferred_lane: return 2
            if lane < self.preferred_lane: return 1
            return 0

    # 3. Initialize Env
    print(f"Initializing Continuous World for {ALGO}...")
    env = gym.make('Continuous-v0', render_mode='human')
    env = PreferredLaneOvertakeWrapper(env)

    # 4. Load Model
    model = load_clean_sb3(PPO if ALGO == "PPO" else DQN, model_path, env)

    if model:
        print(f"\n--- Testing Agent on Continuous World ---")
        try:
            for ep in range(NUM_EPISODES):
                print(f"Episode {ep+1}...")
                obs, _ = env.reset()
                done = False
                
                while not done:
                    # USE THE MODEL HERE
                    action, _ = model.predict(obs, deterministic=True)
                    
                    env.render()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
        except KeyboardInterrupt:
            print("Stopped.")
    
    env.close()