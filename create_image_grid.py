import shutil
import matplotlib.image as mpimg
import wandb
import os
import matplotlib.pyplot as plt

api = wandb.Api()
entity = "replearn"
project = "STM-target-radius-experiment-MNIST"

# # Create a 3x3 grid of subplots for loss curves
# fig, axes = plt.subplots(len(vars0), len(vars1), figsize=(12, 12), sharex=True, sharey=True)
# if mode == "":
#     fig.suptitle("Loss Curves for Different Alpha and Beta Values")
# else:
#     fig.suptitle("Loss Curves for Different Alpha and Beta Values (Vieri Algorithm)")

# # Iterate over vars0 and vars1 to populate each subplot
# for i, alpha in enumerate(vars0):
#     for j, beta in enumerate(vars1):
#         # Define filters for current alpha and beta
#         filters = {
#             "config.BETA": beta,
#             "config.ALPHA": alpha,
#             "config.MODE": mode,
#             "config.SIGMA_BASELINE": sigma_baseline,
#             "state": "finished"
#         }
		
#         # Fetch the runs that match the filters
#         runs = api.runs(
#             path=f"{entity}/{project}",
#             filters=filters,
#             order="-created_at"
#         )
		
#         # Use only the first run that matches the filters
#         for run in runs:
#             # Collect loss data
#             history_losses = run.scan_history(keys=["loss"], page_size=1000, min_step=0, max_step=2000)
#             losses = [row["loss"] for row in history_losses]  # Loss values
#             steps = [i for i in range(len(losses))]  # Step numbers
			
#             # Plot loss on the respective subplot
#             ax = axes[i, j]
#             ax.plot(steps, losses, label=f"alpha={alpha}, beta={beta}")
#             ax.set_title(f"alpha={alpha}, beta={beta}")
#             ax.set_xlabel("Step")
#             ax.set_ylabel("Loss")
#             ax.grid(True)
			
#             # Break after the first matching run
#             break

# # Adjust layout and save the figure
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
# if mode == "":
#     plt.savefig("loss_curves_grid.png", dpi=300)
# else:
#     plt.savefig("loss_curves_grid_Vieri.png", dpi=300)
# plt.show()
# print("Loss curves grid saved")

def create_image_grid(vars0, vars1):
	
	download_dir=".\\downaloads"

	# Create a 3x3 grid of subplots for weight images
	fig, axes = plt.subplots(len(vars0["values"]), len(vars1["values"]), figsize=(12, 12), sharex=True, sharey=True)
	fig.suptitle("Weight Images ("+str(MODE) +")")

	# Iterate over vars0 and vars1 to populate each subplot
	for i, v0 in enumerate(vars0["values"]):
		for j, v1 in enumerate(vars1["values"]):
			# Define filters for current alpha and beta
			filters = {
				"config."+vars0['name'] : v0,
				"config."+vars1['name'] : v1,
				"config.MODE": MODE,
				"state": "finished"
			}
			
			# Fetch the runs that match the filters
			runs = api.runs(
				path=f"{entity}/{project}",
				filters=filters,
				order="-created_at"
			)
			
			# Use only the first run that matches the filters
			for run in runs:
				# Collect weight data at the last step (assuming it's stored in 2D format)
				history_weights = run.scan_history(keys=["weights"], page_size=1000, min_step=0, max_step=2000)

				weights_metadata = [row["weights"] for row in history_weights]

				if weights_metadata:
					# Access the image path for the weights
					weight_img_path = weights_metadata[-1]["path"]
					
					# Download the image file
					run.file(weight_img_path).download(root=download_dir, replace=True)

					# Load the image and display it in the subplot
					weight_img = mpimg.imread(download_dir+"\\"+weight_img_path)
					ax = axes[i, j]
					ax.imshow(weight_img, cmap="viridis")
					ax.set_title(f"{vars0['name']}={v0}, {vars1['name']}={v1}")
					ax.axis("off")  # Hide axes for image display
				# Break after the first matching run
				break
	# Adjust layout and save the figure
	plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
	plt.savefig("weight_images_grid_"+str(MODE)+".png", dpi=300)
	shutil.rmtree(download_dir)
	return

	
MODE="STC-modified"

if __name__=="__main__":
	vars0 = {"values": [5, 4, 3],  # 3 different alpha values
			 "name": "ALPHA"}	  
			
	vars1 = {"values": [10, 5, 2],  # 3 different targed radius values
		  	"name": "target_radius"}	
		
	create_image_grid(vars0, vars1)

	print("Weight images grid created.")