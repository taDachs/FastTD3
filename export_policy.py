# Python snippets for JIT-scripting checkpoints
import torch
from fast_td3 import load_policy
policy = load_policy("./Unitree-Go2-Velocity__FastTD3_Rough_higher_exploration_policy_history_higher_disabled_cmd_curriculum__1_final.pt")
example_inputs = torch.ones(1, policy.n_obs)
# scripted_policy = torch.jit.script(policy)
# scripted_policy.save("./policy.jit")
onnx_program = torch.onnx.export(
    policy, (example_inputs,), input_names=["obs"], output_names=["actions"], dynamo=True
)
onnx_program.optimize()
onnx_program.save("../../policies/fasttd3_rough_policy_history_no_cmd_curr/exported/policy.onnx")
