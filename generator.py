import random
from queue import Queue

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.colors as mcolors
from adjustText import adjust_text
from opensimplex import OpenSimplex
from scipy.ndimage import (
    label,
    center_of_mass,
    zoom,
    gaussian_filter,
    binary_dilation,
)
from scipy.spatial import Delaunay, Voronoi
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
from skimage.measure import regionprops
import plotly.graph_objects as go


matplotlib.rc('font', family='Comic Sans MS')


def generate_perlin_noise_map(shape, scale=100, octaves=15, persistence=0.2, lacunarity=9.0, base=0):
    noise = OpenSimplex(seed=base)
    map_array = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            value = 0
            amplitude = 1
            frequency = 1
            for _ in range(octaves):
                value += amplitude * noise.noise2(i * frequency / scale, j * frequency / scale)
                amplitude *= persistence
                frequency *= lacunarity
            map_array[i][j] = value
    return map_array


def generate_playable_map(n, k):
    scale = random.uniform(80, 120)
    base = random.randint(0, 100)

    # Generate a larger noise map to allow for cropping
    larger_n, larger_k = n + 40, k + 40
    noise_map = generate_perlin_noise_map((larger_n, larger_k), scale=scale, base=base)

    # Crop the noise map to create a water buffer
    noise_map = noise_map[20:-20, 20:-20]

    # Normalize the map
    normalized_map = 2 * ((noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())) - 1

    # Create a gradient mask to ensure water at the edges
    y, x = np.ogrid[:n, :k]
    mask = np.minimum(np.minimum(x, k - 1 - x), np.minimum(y, n - 1 - y)) / 20.0
    mask = np.clip(mask, 0, 1)

    # Apply the mask to the normalized map
    final_map = normalized_map * mask

    return final_map


def land_minimum_calculation(colours, land_colour_names=["#f6cd61", "#fe8a71", "#2a4d69", "#4b86b4"]):
    land_percentage = 0
    for colour in land_colour_names:
        land_percentage += colours.count(colour)
    land_percentage = land_percentage / len(colours)
    min_land_value = np.quantile(np.linspace(-1, 1, 2000), 1 - land_percentage)
    return min_land_value


def land_maximum_calculation(colours, land_colour_names=["#2a4d69", "#4b86b4"]):
    land_percentage = 0
    for colour in land_colour_names:
        land_percentage += colours.count(colour)
    land_percentage = land_percentage / len(colours)
    max_land_value = np.quantile(np.linspace(-1, 1, 2000), 1 - land_percentage)
    return max_land_value


def place_cities(playable_map, colours, number_of_cities):
    n, k = playable_map.shape
    land_indices = []
    land_minimum = land_minimum_calculation(colours)
    land_maximum = land_maximum_calculation(colours)

    for i in range(n):
        for j in range(k):
            if land_minimum < playable_map[i, j] < land_maximum:  # land area (green, yellow, brown)
                land_indices.append((i, j))

    # Calculate proximity to water for each land index
    proximity_scores = []
    for i, j in land_indices:
        distances = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < k:
                    if playable_map[ni, nj] < -1:  # water area
                        distances.append(np.sqrt(di ** 2 + dj ** 2))
        if distances:
            proximity_scores.append(min(distances))
        else:
            proximity_scores.append(np.inf)  # No water nearby

    # Handle case where all proximity_scores are np.inf
    if all(score == np.inf for score in proximity_scores):
        probabilities = np.ones(len(proximity_scores)) / len(proximity_scores)
    else:
        # Normalize proximity scores to probabilities
        proximity_scores = np.array(proximity_scores)
        probabilities = 1 / (proximity_scores + 1)  # Inverse to give higher probability near water
        probabilities[proximity_scores == np.inf] = 0  # Assign zero probability to land with no nearby water
        probabilities /= probabilities.sum()  # Normalize to sum to 1

    # Randomly select city locations based on probabilities
    city_indices = np.random.choice(len(land_indices), size=number_of_cities, p=probabilities, replace=False)
    city_locations = [land_indices[idx] for idx in city_indices]
    return land_minimum, land_maximum, city_locations


def generate_world_name(prefixes, middles, suffixes):
    # Determine the structure of the world name
    structure = random.choice(["prefix-suffix", "prefix-middle-suffix"])

    if structure == "prefix-suffix":
        name = random.choice(prefixes) + random.choice(suffixes)
    else:
        name = random.choice(prefixes) + random.choice(middles) + random.choice(suffixes)

    # Capitalize the first letter and return the name
    return name.capitalize()


def generate_city_name(prefixes, middles, suffixes):
    # Determine the structure of the city name
    structure = random.choice(["prefix-suffix", "prefix-middle-suffix"])

    if structure == "prefix-suffix":
        name = random.choice(prefixes) + random.choice(suffixes)
    else:
        name = random.choice(prefixes) + random.choice(middles) + random.choice(suffixes)

    # Capitalize the first letter and return the name
    return name.capitalize()


def define_countries(playable_map, city_locations, colours, num_countries=5, alpha=0.98, beta=0.02):
    n, k = playable_map.shape
    land_minimum = land_minimum_calculation(colours)  # Ensure this function is defined elsewhere
    land_mask = (playable_map > land_minimum) & (playable_map < 1)

    # Move centers to the closest land point if they're on water
    num_centers = min(num_countries, len(city_locations))
    for i in range(num_centers):
        cy, cx = city_locations[i]
        if not land_mask[cy, cx]:
            # Find the closest land point
            land_points = np.argwhere(land_mask)
            distances = np.sqrt(np.sum((land_points - [cy, cx]) ** 2, axis=1))
            closest_land = land_points[np.argmin(distances)]
            city_locations[i] = tuple(closest_land)

    # Calculate terrain gradients (higher gradient = less desirable)
    gradient_map = np.gradient(playable_map)
    gradient_magnitude = np.sqrt(gradient_map[0] ** 2 + gradient_map[1] ** 2)

    # Initialize country labels (-1 for sea) and country queues
    country_labels = np.full((n, k), -1)
    country_queues = {i: [] for i in range(num_centers)}

    # Place initial city locations and seed their queues
    for country_idx, (cy, cx) in enumerate(city_locations[:num_centers]):
        country_labels[cy, cx] = country_idx
        country_queues[country_idx].append((cy, cx))

    # Directions for polygon expansion (like flood-fill)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # First Pass: Classify territories using dx/dy approach (flood-fill style)
    while any(country_queues.values()):  # While there are still points to process
        for country_idx in range(num_centers):
            if not country_queues[country_idx]:
                continue  # Skip if the queue for this country is empty

            # Randomly pick a point from the queue
            cy, cx = country_queues[country_idx].pop(random.randint(0, len(country_queues[country_idx]) - 1))

            # Expand to neighboring points
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < n and 0 <= nx < k and land_mask[ny, nx] and country_labels[ny, nx] == -1:
                    # Calculate scores for assignment
                    distance = np.sqrt(
                        (ny - city_locations[country_idx][0]) ** 2 + (nx - city_locations[country_idx][1]) ** 2)
                    terrain_penalty = gradient_magnitude[ny, nx]
                    score = alpha * distance + beta * terrain_penalty

                    # Assign to the current country if score is favorable
                    country_labels[ny, nx] = country_idx
                    country_queues[country_idx].append((ny, nx))

    # Step 2: Identify all unclassified land (secondary islands)
    unclassified_mask = (country_labels == -1) & land_mask

    # Step 3: Label unclassified islands
    labeled_map, num_islands = label(unclassified_mask)

    # Step 4: Assign each unclassified island to the country with the most closest points
    for island_id in range(1, num_islands + 1):
        island_mask = (labeled_map == island_id)
        island_points = np.argwhere(island_mask)

        # Calculate distances from each island point to all country centers
        distances = np.array([[np.sqrt(np.sum((point - city) ** 2)) for city in city_locations[:num_centers]]
                              for point in island_points])

        # Find the closest country for each point
        closest_countries = np.argmin(distances, axis=1)

        # Count how many points are closest to each country
        country_counts = np.bincount(closest_countries, minlength=num_centers)

        # Assign the entire island to the country with the most closest points
        dominant_country = np.argmax(country_counts)
        country_labels[island_mask] = dominant_country

    return country_labels  # , city_locations


# Parameters
n, k = 500, 500
number_of_cities = 20  # Number of cities to place
number_of_countries = 7

# Generate the map
playable_map = generate_playable_map(n, k)
zoom_factor = 2  # Define how much you want to upscale
playable_map = zoom(playable_map, zoom_factor, order=4)

# Define the custom colormap
parchment_color = "#dbcfab"
colors_dict = {"#437caf": 53, "#0e9aa7": 42, "#3da4ab": 31, "#f6cd61": 25, "#fe8a71": 30, "#2a4d69": 20, "#4b86b4": 0}
colors = []

for color, count in colors_dict.items():
    colors.extend([color] * count)

n_bins = 2000
cmap_name = "custom_cmap"
cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Place cities
land_minimum, land_maximum, city_locations = place_cities(playable_map, colors, number_of_cities)

# Define lists of syllables for the city and wotld names
prefixes_city = [
    "Al", "Ara", "Bel", "Cal", "Dar", "Eri", "Fen", "Gar", "Hal", "Ira",
    "Kel", "Lan", "Mal", "Nor", "Orin", "Pha", "Quel", "Ren", "Syl", "Tir",
    "Ul", "Var", "Win", "Xan", "Yor", "Zal"
]

prefixes_world = [
    "Ae", "Aur", "Bra", "Cor", "Del", "Elu", "Fal", "Gla", "Hor", "Isen",
    "Jor", "Kai", "Lor", "Mar", "Nor", "Oria", "Pel", "Quor", "Ral", "Sel",
    "Tor", "Ul", "Val", "Xar", "Yla", "Zor"
]

middles_city = [
    "a", "e", "i", "o", "u", "an", "el", "in", "or", "ul",
    "ar", "ia", "ir", "al", "is", "on", "ai", "il", "os", "en"
]

middles_world = [
    "an", "el", "ia", "ir", "on", "ar", "is", "il", "ol", "ul",
    "ai", "eo", "in", "us", "os", "en", "un", "or", "ue", "al"
]

suffixes_city = [
    "ton", "ford", "holm", "dale", "wyn", "fall", "port", "shire", "stead",
    "ward", "lyn", "fell", "mere", "field", "gate", "mont", "haven", "crest"
]

suffixes_world = [
    "ia", "ar", "or", "an", "on", "us", "en", "al", "ur", "is",
    "os", "il", "ea", "ion", "ium", "ith", "oth", "ian", "arn", "orn"
]

# Plot the map with cities
fig = plt.figure(figsize=(12, 8), facecolor=parchment_color)
plt.imshow(playable_map, cmap=cm, interpolation='nearest', alpha=.8)
# plt.colorbar(label="Value")
world_name = generate_world_name(prefixes_world, middles_world, suffixes_world)
plt.title(f"The map of {world_name}", size=max(fig.get_size_inches() * fig.dpi) / 50)
plt.contour(playable_map, levels=7, colors='grey', linewidths=0.5, alpha=.8)
plt.axis('off')  # Hide axes

city_names = []
texts = []
# Overlay cities on the map
for city in city_locations:
    city_name = generate_city_name(prefixes_city, middles_city, suffixes_city)
    plt.scatter(city[1], city[0], color='r', edgecolors='w', s=max(fig.get_size_inches() * fig.dpi) / 30,
                zorder=10)  # 'ro' for red dots
    texts.append(plt.text(city[1], city[0], city_name, color="black", zorder=11))
    city_names.append(city_name)

# Add country borders
country_labels = define_countries(playable_map, city_locations, colors, num_countries=number_of_countries)
centers = []
# texts = []  # Make sure to define the texts list before appending
for i in range(number_of_countries):
    # Create a binary mask for the current country
    classification_matrix = country_labels == i
    cs = plt.contour(classification_matrix, levels=[0.5], linewidths=0.5)

    # Get the center of mass for the classification matrix (country area)
    center = center_of_mass(classification_matrix)

    # Generate a random name for the country
    country_name = generate_world_name(prefixes_world, middles_world, suffixes_world)

    # Add a text annotation at the center, adjusting the font size dynamically based on figure size
    font_size = max(fig.get_size_inches() * fig.dpi) / 100  # Adjusted font size for better readability
    text = plt.text(center[1], center[0], country_name, color="black", size=font_size, zorder=11, ha='center',
                    va='center', weight="bold")

    # Append the created text to the texts list
    texts.append(text)

    # Store the center of each country for potential adjustments
    centers.append(center)

# Adjust the texts to avoid overlap
adjust_text(
    texts,
    only_move={'points': 'xy', 'text': 'xy'},
    arrowprops=dict(arrowstyle="->", color='black', lw=0.7),
    avoid_self=False,  # Set this to True if you want to avoid self-overlap
)

# Tight layout to ensure no clipping
plt.tight_layout()
plt.savefig('my_map')
plt.show()
