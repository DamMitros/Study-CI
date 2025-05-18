import gymnasium as gym, numpy as np
import matplotlib.pyplot as plt
from simpful import *

def create_fuzzy_controller():
  FS = FuzzySystem()
  # Zmienne wejściowe    
  x_position_lv_sets = [
    TriangleFuzzySet(-1.0, -0.5, 0.0, term="left"),
    TriangleFuzzySet(-0.5, 0.0, 0.5, term="center"),
    TriangleFuzzySet(0.0, 0.5, 1.0, term="right")
  ]
  FS.add_linguistic_variable("x_position", LinguisticVariable(x_position_lv_sets, universe_of_discourse=[-1.0, 1.0]))
    
  y_position_lv_sets = [
    TriangleFuzzySet(0.6, 0.8, 1.5, term="high"),
    TriangleFuzzySet(0.4, 0.6, 0.8, term="mid"),
    TriangleFuzzySet(0.2, 0.4, 0.6, term="low"),
    TriangleFuzzySet(-0.2, 0.0, 0.2, term="ground")
  ]
  FS.add_linguistic_variable("y_position", LinguisticVariable(y_position_lv_sets, universe_of_discourse=[-0.2, 1.5]))
    
  x_velocity_lv_sets = [
    TriangleFuzzySet(-1.0, -0.5, 0.0, term="moving_left"),
    TriangleFuzzySet(-0.2, 0.0, 0.2, term="stable"),
    TriangleFuzzySet(0.0, 0.5, 1.0, term="moving_right")
  ]
  FS.add_linguistic_variable("x_velocity", LinguisticVariable(x_velocity_lv_sets, universe_of_discourse=[-1.0, 1.0]))
    
  y_velocity_lv_sets = [
    TriangleFuzzySet(-2.0, -1.5, -1.0, term="falling_fast"),
    TriangleFuzzySet(-1.0, -0.5, 0.0, term="falling"),
    TriangleFuzzySet(-0.2, 0.0, 0.2, term="hovering"),
    TriangleFuzzySet(0.0, 0.5, 1.0, term="rising")
  ]
  FS.add_linguistic_variable("y_velocity", LinguisticVariable(y_velocity_lv_sets, universe_of_discourse=[-2.0, 1.0]))
    
  angle_lv_sets = [
    TriangleFuzzySet(-1.0, -0.5, 0.0, term="tilted_left"),
    TriangleFuzzySet(-0.2, 0.0, 0.2, term="straight"),
    TriangleFuzzySet(0.0, 0.5, 1.0, term="tilted_right")
  ]
  FS.add_linguistic_variable("angle", LinguisticVariable(angle_lv_sets, universe_of_discourse=[-1.0, 1.0]))
  
  # Zmienne wyjściowe
  main_thrust_lv_sets = [
    TriangleFuzzySet(0.0, 0.1, 0.2, term="off"),
    TriangleFuzzySet(0.2, 0.4, 0.6, term="low"),
    TriangleFuzzySet(0.5, 0.65, 0.8, term="medium"),
    TriangleFuzzySet(0.7, 0.85, 1.0, term="high"),
    TriangleFuzzySet(0.9, 0.95, 1.0, term="very_high")  
  ]
  FS.add_linguistic_variable("main_thrust", LinguisticVariable(main_thrust_lv_sets, universe_of_discourse=[0.0, 1.0]))
    
  lateral_thrust_lv_sets = [
    TriangleFuzzySet(-1.0, -1.0, -0.8, term="very_high_left"),
    TriangleFuzzySet(-1.0, -0.75, -0.5, term="high_left"),
    TriangleFuzzySet(-0.6, -0.4, -0.2, term="medium_left"),
    TriangleFuzzySet(-0.1, 0.0, 0.1, term="neutral"),
    TriangleFuzzySet(0.2, 0.4, 0.6, term="medium_right"),
    TriangleFuzzySet(0.5, 0.75, 1.0, term="high_right"),
    TriangleFuzzySet(0.8, 1.0, 1.0, term="very_high_right") 
  ]
  FS.add_linguistic_variable("lateral_thrust", LinguisticVariable(lateral_thrust_lv_sets, universe_of_discourse=[-1.0, 1.0]))
  
  # Reguły
  rules_main = [
    "IF (y_position IS high) AND (y_velocity IS falling) THEN (main_thrust IS low)",
    "IF (y_position IS high) AND (y_velocity IS falling_fast) THEN (main_thrust IS medium)",
    "IF (y_position IS mid) AND (y_velocity IS falling) THEN (main_thrust IS medium)",
    "IF (y_position IS mid) AND (y_velocity IS falling_fast) THEN (main_thrust IS high)",
    "IF (y_position IS low) AND (y_velocity IS falling) THEN (main_thrust IS high)",
    "IF (y_position IS low) AND (y_velocity IS falling_fast) THEN (main_thrust IS very_high)",
    "IF (y_position IS ground) AND (y_velocity IS falling) THEN (main_thrust IS very_high)",
    "IF (y_position IS ground) AND (y_velocity IS falling_fast) THEN (main_thrust IS very_high)",
    "IF (y_velocity IS hovering) AND (y_position IS high) THEN (main_thrust IS low)",
    "IF (y_velocity IS hovering) AND (y_position IS mid) THEN (main_thrust IS medium)",
    "IF (y_velocity IS hovering) AND (y_position IS low) THEN (main_thrust IS medium)",
    "IF (y_velocity IS hovering) AND (y_position IS ground) THEN (main_thrust IS off)",
    "IF (y_velocity IS rising) THEN (main_thrust IS off)",
    "IF (y_position IS high) AND (y_velocity IS rising) THEN (main_thrust IS off)",
  ]
    
  rules_lateral = [
    "IF (x_position IS right) AND (x_velocity IS moving_right) THEN (lateral_thrust IS high_left)",
    "IF (x_position IS right) AND (x_velocity IS stable) THEN (lateral_thrust IS medium_left)",
    "IF (x_position IS center) AND (x_velocity IS moving_right) THEN (lateral_thrust IS medium_left)",
    "IF (x_position IS center) AND (x_velocity IS stable) THEN (lateral_thrust IS neutral)",
    "IF (x_position IS center) AND (x_velocity IS moving_left) THEN (lateral_thrust IS medium_right)",
    "IF (x_position IS left) AND (x_velocity IS stable) THEN (lateral_thrust IS medium_right)",
    "IF (x_position IS left) AND (x_velocity IS moving_left) THEN (lateral_thrust IS very_high_right)",
    "IF (angle IS tilted_right) THEN (lateral_thrust IS medium_left)",
    "IF (angle IS tilted_left) THEN (lateral_thrust IS medium_right)"
  ]
  FS.add_rules(rules_main + rules_lateral)
    
  return FS

def plot_fuzzy_variables(FS):
  input_vars = ["x_position", "y_position", "x_velocity", "y_velocity", "angle"]
  output_vars = ["main_thrust", "lateral_thrust"]
    
  for var_name in input_vars + output_vars:
    var = FS._lvs[var_name]
    plt.figure(figsize=(10, 5))
    plt.title(f"Zmienna lingwistyczna: {var_name}")
    plt.xlabel("Wartość")
    plt.ylabel("Stopień przynależności")
        
    uod = var.get_universe_of_discourse()
    if uod and len(uod) == 2:
      x_min, x_max = uod
    elif var_name in output_vars:
      x_min, x_max = -1.0, 1.0
    else:
      x_min, x_max = -2.0, 2.0
      if var_name == "y_position":
        x_min, x_max = -0.2, 1.5
        
    x = np.linspace(x_min, x_max, 1000)
        
    for fuzzy_set in var._FSlist:
      term_name = fuzzy_set.get_term()
      y = [fuzzy_set.get_value(val) for val in x]
      plt.plot(x, y, label=term_name)
        
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"fuzzy_variable_{var_name}.png")

def normalize_state(state):
  x, y, vx, vy, angle, _, _, _ = state
    
  x = np.clip(x, -1.0, 1.0)
  y = np.clip(y, -0.2, 1.5)
  vx = np.clip(vx / 2.0, -1.0, 1.0)
  vy = np.clip(vy / 2.0, -2.0, 1.0)
  angle = np.clip(angle / np.pi, -1.0, 1.0)
    
  return {
    "x_position": x,
    "y_position": y,
    "x_velocity": vx,
    "y_velocity": vy,
    "angle": angle
  }

def main():
  env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    
  FS = create_fuzzy_controller()
  plot_fuzzy_variables(FS)
    
  state, _ = env.reset()
  total_reward = 0
    
  for t in range(1000):
    normalized_state = normalize_state(state)    
    for var_name, var_value in normalized_state.items():
      FS.set_variable(var_name, var_value)
        
    result = FS.inference()
        
    # Failsafe: jeśli lądownik jest za wysoko i rośnie dalej, wyłącz thrust
    if normalized_state["y_position"] > 1.2 and normalized_state["y_velocity"] > 0:
      print(f"🛑 Failsafe aktywowany: y={normalized_state['y_position']:.2f}, vy={normalized_state['y_velocity']:.2f}")
      result["main_thrust"] = 0.0
        
    action = np.array([result["main_thrust"], result["lateral_thrust"]])  
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
        
    if terminated or truncated:
      print(f"Epoka zakończona po {t+1} krokach, suma nagród: {total_reward:.2f}")
      break
    
  env.close()

main()