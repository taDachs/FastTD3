# Python snippets for JIT-scripting checkpoints
import sys
import torch
from fast_td3 import load_policy
pt_file = sys.argv[1]
export_file = sys.argv[2]
policy = load_policy(pt_file)
example_inputs = torch.ones(1, policy.n_obs)
# scripted_policy = torch.jit.script(policy)
# scripted_policy.save("./policy.jit")
onnx_program = torch.onnx.export(
    policy, (example_inputs,), input_names=["obs"], output_names=["actions"], dynamo=True
)
onnx_program.optimize()
onnx_program.save(export_file)
