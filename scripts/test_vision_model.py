import torch
from tensordict import TensorDict, is_tensor_collection
from fast_td3.vision_backbone import DepthOnlyFCBackbone58x87
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

teacher_dict: TensorDict = TensorDict.load_memmap("../../stored_transitions/")

vision_obs = teacher_dict["vision_observations"]

print("images:")
print("\tmax: ", vision_obs.max())
print("\tmin: ", vision_obs.min())
print("\tmean: ", vision_obs.mean())
print("---")


sd = torch.load("../../model-with-normalized-vision-and-new-activation.pt", map_location="cpu", weights_only=False)
cleaned_state_dict = {}
bad_name = "_orig_mod."

for k, v in sd["vision_model"].items():
    if bad_name in k:
        cleaned_state_dict[k[len(bad_name):]] = v
    else:
        cleaned_state_dict[k] = v
vision_nn = DepthOnlyFCBackbone58x87(128)
vision_nn.load_state_dict(cleaned_state_dict)

print(vision_obs.shape)
sample = vision_obs[:, 500:550].flatten(0, 1)
latent = vision_nn(sample.permute(0, 3, 1, 2) / 10.0).detach().cpu().numpy()
print("latent:")
print("\tmax: ", latent.max())
print("\tmin: ", latent.min())
print("\tmean: ", latent.mean())
print("---")

pca = PCA(2)
y = pca.fit_transform(latent)
print(y)


plt.scatter(y[:, 0], y[:, 1])
plt.show()
