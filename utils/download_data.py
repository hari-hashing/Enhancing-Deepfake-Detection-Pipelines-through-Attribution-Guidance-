import kagglehub

# Download latest version
path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")

print(f"the path of the downloaded dataset is: {path}")