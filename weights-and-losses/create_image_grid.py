import shutil
import matplotlib.image as mpimg
import wandb
import os
import matplotlib.pyplot as plt
import pandas

api = wandb.Api()
entity = "replearn"
project = "STM-target-radius-experiment-MNIST"

import matplotlib.pyplot as plt
import pandas

def create_loss_grid(vars0: list, vars1: list, loss_name: str, modes: list = ["Base", "Base_Norm", "STC-modified"]):
    df = pandas.read_csv(".\\loss_results_table.csv")
    fig, axes = plt.subplots(len(vars0["values"]), len(vars1["values"]), figsize=(12, 12), sharex=True, sharey=True)
    fig.suptitle(loss_name)

    # Initialize variables to store legend handles and labels
    legend_handles = []
    labels = modes

    for i, v0 in enumerate(vars0["values"]):
        for j, v1 in enumerate(vars1["values"]):
            x_values = [0, 1, 2]
            y_values = [df[(df[vars0['name']]==v0) & (df[vars1['name']]==v1) & (df["MODE"]==modes[0]) ][loss_name].values[0],
			   			df[(df[vars0['name']]==v0) & (df[vars1['name']]==v1) & (df["MODE"]==modes[1])][loss_name].values[0],
						df[(df[vars0['name']]==v0) & (df[vars1['name']]==v1) & (df["MODE"]==modes[2])][loss_name].values[0]]							


            ax = axes[i, j]
            ax.set_title(f"{vars0['name']}={v0}, {vars1['name']}={v1}")
            bars = ax.bar(x_values, y_values, color=['blue', 'red', 'green'])
            ax.grid(True)

            # Capture the first instance of bars to use for the legend
            if i == 0 and j == 0:
                legend_handles = bars  # Store the bars from the first subplot

    # Add a single legend for the entire figure
    fig.legend(legend_handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(".\\weights-and-losses\\" + loss_name + ".png", dpi=300)
    return

	

def create_image_grid(vars0, vars1):
	
	download_dir=".\\weights-images\\downaloads"

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
	plt.savefig(".\\weights-and-losses\\weight_images_grid_"+str(MODE)+".png", dpi=300)
	shutil.rmtree(download_dir)
	return

	
MODE="Base_Norm"

if __name__=="__main__":
	vars0 = {"values": [5, 1, 0.2, 0],  # 3 different alpha values
			 "name": "ALPHA"}	  
			
	vars1 = {"values": [10, 5, 2, 1.5],  # 3 different targed radius values
		  	"name": "target_radius"}	
		
	#create_image_grid(vars0, vars1)
	create_loss_grid(vars0, vars1, "distance_target_bmu")
	print("Image created.")