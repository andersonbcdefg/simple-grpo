import numpy as np
import matplotlib.pyplot as plt
import os
import math

def generate_correlation_plot(r_value, num_points=75, filename="correlation_plot.png", img_size=(224, 224)):
    """
    Generates a scatter plot image for a specified Pearson correlation coefficient R.

    Args:
        r_value (float): The desired Pearson correlation coefficient (between 0 and 1).
        num_points (int): The number of points in the scatter plot. Defaults to 75.
        filename (str): The path to save the output PNG image.
        img_size (tuple): The desired output image size in pixels (width, height).
    """
    if not (0 <= r_value <= 1):
        raise ValueError("r_value must be between 0 and 1")

    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir: # Check if dirname is not empty (i.e., not saving in root)
        os.makedirs(output_dir, exist_ok=True)

    # Generate figure and axes for plotting
    # Use figure size and DPI to control the output pixel size
    dpi = 100
    fig, ax = plt.subplots(figsize=(img_size[0]/dpi, img_size[1]/dpi), dpi=dpi)

    # Handle the edge case R=1 separately to avoid sqrt of negative number or division by zero in standardization
    if math.isclose(r_value, 1.0):
        # For R=1, y should just be x (or a scaled version)
        x_independent = np.random.randn(num_points) # Standard normal distribution
        x = (x_independent - np.mean(x_independent)) / np.std(x_independent) if np.std(x_independent) != 0 else x_independent
        y = x # Perfect correlation
    elif math.isclose(r_value, 0.0):
        # For R=0, x and y are independent
        x_independent = np.random.randn(num_points)
        y_independent = np.random.randn(num_points)
        x = (x_independent - np.mean(x_independent)) / np.std(x_independent) if np.std(x_independent) != 0 else x_independent
        y = (y_independent - np.mean(y_independent)) / np.std(y_independent) if np.std(y_independent) != 0 else y_independent
    else:
        # 1. Generate independent standard normal data
        mean = [0, 0]
        cov = [[1, 0], [0, 1]] # Diagonal covariance matrix for independence
        x_independent, y_independent = np.random.multivariate_normal(mean, cov, num_points).T

        # Standardize explicitly for robustness, handling zero standard deviation case
        x_std = np.std(x_independent)
        y_prime_std = np.std(y_independent)

        x = (x_independent - np.mean(x_independent)) / x_std if x_std != 0 else x_independent
        y_prime = (y_independent - np.mean(y_independent)) / y_prime_std if y_prime_std != 0 else y_independent

        # 2. Introduce correlation: y = r*x + sqrt(1-r^2)*y_prime
        y = r_value * x + math.sqrt(1 - r_value**2) * y_prime
        # Re-standardize y to have similar scale visually, though correlation is already set
        y_std = np.std(y)
        y = (y - np.mean(y)) / y_std if y_std != 0 else y


    # 3. Plot points - use small black squares like the example
    ax.scatter(x, y, color='black', marker='s', s=10, zorder=1) # s is marker size, zorder=1 to plot above axes

    # Add light grey axes lines
    ax.axhline(0, color='lightgrey', linewidth=1, zorder=0) # Draw behind points
    ax.axvline(0, color='lightgrey', linewidth=1, zorder=0) # Draw behind points

    # Customize appearance to match example (minimalist)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Set limits to ensure points are centered and prevent points touching edge
    # Calculate dynamic limits based on data range plus padding
    all_vals = np.concatenate((x, y))
    if all_vals.size > 0:
        lim_min = np.min(all_vals)
        lim_max = np.max(all_vals)
        padding = (lim_max - lim_min) * 0.1 # Add 10% padding
        ax.set_xlim(lim_min - padding, lim_max + padding)
        ax.set_ylim(lim_min - padding, lim_max + padding)
    else:
        # Default limits if no data (should not happen with num_points > 0)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)


    # Remove padding around the plot itself
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator()) 

    # 4. Save
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig) # Close the figure to free memory

if __name__ == "__main__":
    # Example usage: Generate plots similar to the user's example image
    output_base_dir = "generated_datasets"
    dataset_name = "correlation_scatter"
    output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Define the specific R values from the example
    example_r_values = [0.0, 0.5, 0.75, 0.90, 0.95, 1.0]
    num_points_per_plot = 75
    image_dimensions = (224, 224) # Standard size

    print(f"Generating example correlation plots in '{output_dir}'...")

    for r in example_r_values:
        # Format filename to match R value precisely
        r_str = f"{r:.2f}".replace('.', '_') # e.g., 0_50, 1_00
        fname = os.path.join(output_dir, f"correlation_R_{r_str}.png")
        print(f"  Generating plot for R={r:.2f} -> {fname}")
        try:
            generate_correlation_plot(
                r_value=r,
                num_points=num_points_per_plot,
                filename=fname,
                img_size=image_dimensions
            )
        except ValueError as e:
            print(f"    Error generating plot for R={r}: {e}")
        except Exception as e:
             print(f"    Unexpected error generating plot for R={r}: {e}")


    # Generate one plot with a random R value as well
    random_r = np.random.rand() # Random float between 0.0 and 1.0
    r_str_random = f"{random_r:.2f}".replace('.', '_')
    fname_random = os.path.join(output_dir, f"correlation_R_{r_str_random}_random.png")
    print(f"  Generating plot for random R={random_r:.2f} -> {fname_random}")
    try:
        generate_correlation_plot(
            r_value=random_r,
            num_points=num_points_per_plot,
            filename=fname_random,
            img_size=image_dimensions
        )
    except Exception as e:
            print(f"    Unexpected error generating plot for random R={random_r}: {e}")


    print(f"Finished generating example plots.")
    print(f"You can find them in the directory: {os.path.abspath(output_dir)}") 