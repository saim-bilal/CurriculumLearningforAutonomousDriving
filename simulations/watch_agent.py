import sys
import warnings
import zipfile
import io
import types
import numpy as np
import torch
import gymnasium as gym
import highway_env  # <--- FIXED: Added missing import
from stable_baselines3 import PPO, DQN

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ALGO = "DQN"           # Options: "PPO", "DQN", "SimpleDQN"
REGIMEN = "curriculum" # Options: "curriculum", "noncurriculum"
NUM_EPISODES = 5       # How many episodes to watch

# ==============================================================================
# COMPATIBILITY PATCH
# ==============================================================================
try:
    import numpy._core
except ImportError:
    # Spoof numpy._core for pickle compatibility
    core_stub = types.ModuleType("numpy._core")
    core_stub.numeric = np.core.numeric
    core_stub.multiarray = np.core.multiarray
    core_stub.umath = np.core.umath
    sys.modules["numpy._core"] = core_stub
    sys.modules["numpy._core.numeric"] = np.core.numeric
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.umath"] = np.core.umath

warnings.filterwarnings("ignore")

# ==============================================================================
# WRAPPER DEFINITION (Crucial for Action Space Match: 7 Actions)
# ==============================================================================
class PreferredLaneOvertakeWrapper(gym.Wrapper):
    def __init__(self, env, preferred_lane=1, speed_limit=33.0):
        super().__init__(env)
        self.preferred_lane = preferred_lane
        self.speed_limit = speed_limit
        self._base_actions = {
            "IDLE": 0, "LANE_LEFT": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4
        }
        # This defines the 7 actions the model expects
        self.macro_actions = [
            {"base": self._base_actions["IDLE"]},
            {"base": self._base_actions["LANE_LEFT"]},
            {"base": self._base_actions["LANE_RIGHT"]},
            {"base": self._base_actions["FASTER"]},
            {"base": self._base_actions["SLOWER"]},
            {"base": self._base_actions["SLOWER"]},
            {"kind": "return"}
        ]
        self.action_space = gym.spaces.Discrete(len(self.macro_actions))

    def step(self, action):
        macro_idx = int(action) % len(self.macro_actions)
        macro = self.macro_actions[macro_idx]
        base_action = macro.get("base")
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

# ==============================================================================
# ROBUST LOADER
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
# SIMULATION SETUP
# ==============================================================================
def make_eval_env():
    # 1. Create Base Env
    env = gym.make("highway-v0", render_mode="human")
    
    # 2. Configure for Live Watching
    config = env.unwrapped.config
    config["duration"] = 60
    config["vehicles_density"] = 1.0
    config["simulation_frequency"] = 15
    config["policy_frequency"] = 5
    config["screen_width"] = 1000
    config["screen_height"] = 600
    config["centering_position"] = [0.3, 0.5]
    config["scaling"] = 6.0
    
    # 3. Apply the Wrapper (Fixes the 5 vs 7 action mismatch)
    env = PreferredLaneOvertakeWrapper(env, preferred_lane=1)
    
    return env

if __name__ == "__main__":
    # 1. Locate File
    base_path = f"experiments/{ALGO}/{REGIMEN}/seed_0"
    import glob
    import os
    run_dirs = glob.glob(os.path.join(base_path, "run_*"))
    if not run_dirs:
        print(f"No run found for {ALGO}/{REGIMEN} in {base_path}")
        sys.exit()
    
    latest_run = max(run_dirs, key=os.path.getmtime)
    model_path = os.path.join(latest_run, "final_model.zip")
    
    if not os.path.exists(model_path):
        print(f"final_model.zip not found in {latest_run}")
        sys.exit()

    # 2. Create Environment
    env = make_eval_env()

    # 3. Load Model
    model = load_clean_sb3(PPO if ALGO == "PPO" else DQN, model_path, env)

    # 4. Watch Loop
    if model:
        print(f"\n--- Watching {ALGO} ({REGIMEN}) ---")
        print(f"Press Ctrl+C in terminal to stop.")
        
        try:
            for ep in range(NUM_EPISODES):
                obs, info = env.reset()
                done = False
                truncated = False
                score = 0
                step = 0
                
                print(f"Episode {ep+1} started...")
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    score += reward
                    step += 1
                
                print(f"Episode {ep+1} finished. Steps: {step}, Score: {score:.2f}")
        except KeyboardInterrupt:
            print("\nStopped by user.")
    
    env.close()