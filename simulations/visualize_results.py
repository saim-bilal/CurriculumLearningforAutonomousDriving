import sys
import warnings
import zipfile
import pickle
import json
import io
import types
from typing import Any, Dict

# ==============================================================================
# 0. COMPATIBILITY PATCH & SETUP
# ==============================================================================
import numpy as np
print(f"DEBUG: Detected NumPy version: {np.__version__}")

# Patch numpy._core if missing (NumPy 2.x -> 1.x compat)
try:
    import numpy._core
except ImportError:
    # Spoof numpy._core modules for the unpickler
    core_stub = types.ModuleType("numpy._core")
    core_stub.numeric = np.core.numeric
    core_stub.multiarray = np.core.multiarray
    core_stub.umath = np.core.umath
    sys.modules["numpy._core"] = core_stub
    sys.modules["numpy._core.numeric"] = np.core.numeric
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.umath"] = np.core.umath

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import random
import glob
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO, DQN
from gymnasium.wrappers import RecordVideo

# ==============================================================================
# 1. Custom Robust Loader
# ==============================================================================

class NumPy2CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy.random._pcg64" and name == "PCG64":
            import numpy.random
            return numpy.random.PCG64
        if module == "numpy._core.numeric":
            return np.core.numeric
        if module == "numpy._core.multiarray":
            return np.core.multiarray
        return super().find_class(module, name)

def load_robust_sb3(algo_class, path, env):
    """
    Robustly loads an SB3 model, handling NumPy version mismatches and JSON/Pickle confusion.
    """
    print(f"Loading {algo_class.__name__} via Robust Loader from {path}...")
    
    data = None
    params = None
    
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # 1. Try to load 'data' (Metadata/Config)
            try:
                with zf.open("data") as f:
                    file_content = f.read()
                    try:
                        # Try Pickle first
                        unpickler = NumPy2CompatUnpickler(io.BytesIO(file_content))
                        data = unpickler.load()
                    except Exception:
                        # Fallback to JSON if pickle fails (invalid load key '{')
                        data = json.loads(file_content.decode('utf-8'))
                        print(f"DEBUG: Loaded 'data' as JSON.")
            except Exception as e:
                print(f"DEBUG: Could not load metadata 'data': {e}")
                data = {}

            # 2. Load 'policy.pth' (Weights)
            with zf.open("policy.pth") as f:
                content = io.BytesIO(f.read())
                params = torch.load(content, map_location="cpu")

        # 3. Strategy A: If we have data, try to use it (Standard Path)
        if data and "policy_class" in data:
            try:
                model = algo_class(
                    policy=data["policy_class"],
                    env=env,
                    device="cpu",
                    _init_setup_model=False
                )
                model.__dict__.update(data)
                model.observation_space = env.observation_space
                model.action_space = env.action_space
                model.set_random_seed(0)
                model._setup_model()
                model.policy.load_state_dict(params)
                print(f"{algo_class.__name__} loaded successfully (Standard).")
                return model
            except Exception as e:
                print(f"Standard load failed ({e}). Attempting Blind Load...")

        # 4. Strategy B: Blind Load (Ignore metadata, force weights)
        # Try Default Architecture first
        print("Attempting Blind Load (Default Architecture)...")
        try:
            model = algo_class("MlpPolicy", env=env, device="cpu")
            model.policy.load_state_dict(params)
            print(f"{algo_class.__name__} loaded successfully (Blind - Default).")
            return model
        except RuntimeError as e:
            print(f"Default arch failed. Trying custom [256, 256]...")
        
        # Try Custom Architecture [256, 256] (Common in this course)
        try:
            policy_kwargs = dict(net_arch=[256, 256])
            # For DQN, net_arch is passed differently sometimes, but usually via policy_kwargs
            if algo_class == DQN:
                 # DQN uses 'net_arch' inside policy_kwargs directly or just net_arch
                 # But SB3 DQN MlpPolicy takes net_arch in policy_kwargs
                 pass
            
            model = algo_class("MlpPolicy", env=env, policy_kwargs=policy_kwargs, device="cpu")
            model.policy.load_state_dict(params)
            print(f"{algo_class.__name__} loaded successfully (Blind - [256, 256]).")
            return model
        except Exception as e:
            print(f"All load strategies failed for {algo_class.__name__}: {e}")
            return None

    except Exception as e:
        print(f"Robust Load Failed completely: {e}")
        return None

# ==============================================================================
# 2. Environment & Wrapper Definitions
# ==============================================================================

def _create_single_env(map_name, traffic_density, use_discrete=True, seed=None):
    env = gym.make(map_name, render_mode="rgb_array")
    
    config = env.unwrapped.config
    config["simulation_frequency"] = 15
    config["policy_frequency"] = 5
    config["duration"] = 40
    config["vehicles_density"] = traffic_density
    config["vehicles_count"] = max(5, int(50 * traffic_density))
    
    config["screen_width"] = 600
    config["screen_height"] = 600
    config["centering_position"] = [0.3, 0.5]
    config["scaling"] = 5.5
    
    discrete_actions = use_discrete or ("merge" in map_name)
    if discrete_actions:
        config["action"] = {"type": "DiscreteMetaAction"}
    
    config["observation"] = {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,
        "absolute": False,
    }
    config["offscreen_rendering"] = True
    
    if "highway" in map_name:
        config["lanes_count"] = 4
    if "roundabout" in map_name:
        config["incoming_vehicle_destination"] = None
    
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    else:
        env.reset()
    return env

class StitchedScenarioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, block_specs, env_builder, stage_label="MultiBlock", shuffle_blocks=False):
        super().__init__()
        self.block_specs = block_specs
        self.env_builder = env_builder
        self.stage_label = stage_label
        self.shuffle_blocks = shuffle_blocks
        self._rng = random.Random()
        self.block_envs = [self.env_builder(spec, idx) for idx, spec in enumerate(block_specs)]
        self.blocks_total = len(self.block_envs)
        self.block_order = list(range(self.blocks_total))
        self.current_block_pointer = 0
        self.current_env = self.block_envs[0]
        self.current_spec = self.block_specs[0]
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space
        self.render_mode = "rgb_array"

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        self.block_order = list(range(self.blocks_total))
        if self.shuffle_blocks:
            self._rng.shuffle(self.block_order)
        self.current_block_pointer = 0
        block_id = self.block_order[self.current_block_pointer]
        self.current_env = self.block_envs[block_id]
        self.current_spec = self.block_specs[block_id]
        obs, info = self._reset_block(block_id, seed)
        return obs, self._augment_info(info, False)

    def step(self, action):
        step_out = self.current_env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        else:
            obs, reward, done, info = step_out
            terminated = bool(done)
            truncated = False
        
        transition = False
        episode_done = terminated or truncated
        
        if episode_done:
            next_pointer = self.current_block_pointer + 1
            if next_pointer < self.blocks_total:
                transition = True
                self.current_block_pointer = next_pointer
                block_id = self.block_order[self.current_block_pointer]
                self.current_env = self.block_envs[block_id]
                self.current_spec = self.block_specs[block_id]
                obs, info = self._reset_block(block_id)
                terminated = False
                truncated = False
                
        info = self._augment_info(info, transition)
        return obs, reward, terminated, truncated, info

    def _reset_block(self, block_id, base_seed=None):
        if base_seed is not None:
            block_seed = base_seed + block_id * 101
        else:
            block_seed = self._rng.randint(0, 1_000_000)
        
        try:
            reset_out = self.block_envs[block_id].reset(seed=block_seed)
        except TypeError:
            if block_seed is not None:
                self.block_envs[block_id].seed(block_seed)
            reset_out = self.block_envs[block_id].reset()
            
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            return reset_out
        return reset_out, {}

    def _augment_info(self, info, transition):
        info = info or {}
        if isinstance(info, list) and info:
            info = info[0]
        info.update({
            "block_name": self.current_spec.get("label", self.current_spec.get("map", "block")),
            "block_transition": transition
        })
        return info

    def render(self):
        return self.current_env.render()

    def close(self):
        for env in self.block_envs:
            env.close()

class PreferredLaneOvertakeWrapper(gym.Wrapper):
    def __init__(self, env, preferred_lane=1, speed_limit=33.0):
        super().__init__(env)
        self.preferred_lane = preferred_lane
        self.speed_limit = speed_limit
        self._base_actions = {
            "IDLE": 0, "LANE_LEFT": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4
        }
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
# 3. SimpleDQN Agent Definition
# ==============================================================================

class SimpleQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, net_arch):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class SimpleDQNAgent:
    def __init__(self, env, config, device='cpu'):
        self.env = env
        self.device = torch.device(device)
        self.obs_shape = env.observation_space.shape
        self.obs_dim = int(np.prod(self.obs_shape))
        self.action_dim = env.action_space.n
        self.q_net = SimpleQNetwork(self.obs_dim, self.action_dim, config['net_arch']).to(self.device)

    def predict(self, observation, deterministic=True):
        obs_tensor = torch.from_numpy(observation.reshape(1, -1)).float().to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action, None

    @classmethod
    def load(cls, path, env):
        print(f"Loading SimpleDQN from {path}...")
        try:
            payload = torch.load(path, map_location='cpu')
            config = payload.get('config', {'net_arch': [256, 256]}) 
            agent = cls(env, config)
            agent.q_net.load_state_dict(payload['q_net'])
            print("SimpleDQN loaded successfully.")
            return agent
        except Exception as e:
            print(f"Failed to load SimpleDQN: {e}")
            return None

# ==============================================================================
# 4. Visualization Setup
# ==============================================================================

def make_eval_env(scenario_spec):
    def block_builder(block_spec, block_idx):
        return _create_single_env(
            map_name=block_spec["map"],
            traffic_density=block_spec.get("traffic", 0.3),
            use_discrete=True,
            seed=None
        )
    
    env = StitchedScenarioEnv(
        block_specs=scenario_spec["blocks"],
        env_builder=block_builder,
        stage_label=scenario_spec.get("label", "Viz"),
        shuffle_blocks=False
    )
    env = PreferredLaneOvertakeWrapper(env, preferred_lane=1)
    return env

def find_model_file(algo, regimen, seed=0):
    base_path = f"experiments/{algo}/{regimen}/seed_{seed}"
    run_dirs = glob.glob(os.path.join(base_path, "run_*"))
    if not run_dirs:
        return None
    latest_run = max(run_dirs, key=os.path.getmtime)
    return os.path.join(latest_run, "final_model.zip")

def record_agent(model, env, name, length=500):
    video_folder = "videos"
    os.makedirs(video_folder, exist_ok=True)
    
    env = RecordVideo(
        env, 
        video_folder=video_folder, 
        name_prefix=name,
        episode_trigger=lambda x: True,
        disable_logger=True
    )
    
    print(f"Recording {name}...")
    try:
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated) and step < length:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            step += 1
        
        env.close()
        print(f"Finished recording {name} ({step} steps).")
    except Exception as e:
        print(f"Error during recording {name}: {e}")
        env.close()

# ==============================================================================
# 5. Main Execution
# ==============================================================================

if __name__ == "__main__":
    STAGE_4_SPEC = {
        "type": "multi_block",
        "label": "Stage4_Complex",
        "blocks": [
            {"map": "highway-v0", "traffic": 0.35},
            {"map": "merge-v0", "traffic": 0.40},
            {"map": "intersection-v0", "traffic": 0.40},
            {"map": "roundabout-v0", "traffic": 0.45}
        ]
    }

    ALGORITHMS = ["PPO", "DQN", "SimpleDQN"]
    REGIMENS = ["curriculum", "noncurriculum"]

    for algo in ALGORITHMS:
        for regimen in REGIMENS:
            print(f"\n--- Processing {algo} ({regimen}) ---")
            
            try:
                env = make_eval_env(STAGE_4_SPEC)
            except Exception as e:
                print(f"Error creating environment: {e}")
                continue
            
            model_path = find_model_file(algo, regimen)
            if not model_path or not os.path.exists(model_path):
                print(f"Skipping: Model file not found at {model_path}")
                env.close()
                continue
            
            model = None
            try:
                if algo == "SimpleDQN":
                    model = SimpleDQNAgent.load(model_path, env)
                elif algo == "PPO":
                    model = load_robust_sb3(PPO, model_path, env)
                elif algo == "DQN":
                    model = load_robust_sb3(DQN, model_path, env)
            except Exception as e:
                print(f"Error loading model: {e}")
            
            if model:
                video_name = f"{algo}_{regimen}"
                record_agent(model, env, video_name)
            else:
                env.close()

    print("\nDone! Check the 'videos/' folder.")