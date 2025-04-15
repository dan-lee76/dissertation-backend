# Generative Pathfinding for UK Hiking Routes

This project implements an A* pathfinding algorithm to generate optimal walking routes, focusing on fitness, distance, and peak inclusion. It uses OpenStreetMap (OSM) data via `osmnx` to build a graph of walkable paths and supports two modes: normal route generation and peak-bagger route generation for maximizing peak visits within a target distance.

The project is part of a dissertation to demonstrate advanced pathfinding techniques for outdoor route planning, with applications in hiking and recreational navigation.

## Features
- **Custom A Star Algorithm**: Generates routes optimizing for fitness (difficulty), distance, and optionally peak visits.
- **Two Modes**:
  - **Normal Generation**: Finds routes within a specified distance range.
  - **Peak-Bagger Generation**: Prioritizes routes that include peaks, ideal for hikers aiming to visit summits.
- **Flexible Inputs**: Specify start/end coordinates, target distance, and tolerance.
- **Output Options**: Exports as HTML or GPX files.
- **Accessibility Considerations**: Incorporates highway and surface scores to favor accessible paths.

## Requirements
- Python 3.12.3
- Conda (recommended for environment management)
- Dependencies listed in `environment.yml` or `requirements.txt`

## Installation

### Using Conda (Recommended)
1. Ensure [Conda](https://docs.conda.io/en/latest/) is installed.
2. 
```bash
conda env create -f environment.yml 
conda activate dissertation-backend
```

### Using Pip (Easier)
1. `pip install -r requirements.txt`


## Usage
The project provides a command-line interface (CLI) to generate and visualize routes. Run main.py with appropriate arguments to configure your route. Inital running to build graph can take time due to API rate limit.

Command-Line Interface

`python main.py --start LAT,LON --end LAT,LON --distance DISTANCE --tolerance TOLERANCE [OPTIONS]`

Required Arguments:
- `--start LAT,LON`: Starting coordinates (e.g., 53.364861,-1.816006).
- `--distance DISTANCE`: Target distance in kilometers (e.g., 10).

Optional Arguments
- `--end LAT,LON`: Ending coordinates (e.g., 53.343444,-1.778107).
- `--tolerance TOLERANCE`: Distance tolerance in kilometers (default: 2.0).
- `--simplify`: Simplify the graph.
- `--mode {normal,peak-bagger}`: Route generation mode (default: peak-bagger).
- `--all-peaks`: Include all peaks in the route).
- `--worst-fitness`: Select route with the worst fitness score.
- `--peak-fitness`: Change how the program will select the route based on peak fitness. Options: 'most', 'least', 'ignore' (default: 'most').
- `--closest-distance`: Select route closest to the target distance (default: False).
- `--favour-descent`: Nodes to use in normal generation: 'all' or 'top10' (default: 'all')
- `--fitnes-mode`: Customise the fitness used in path generation. Use config.py to modify custom fitness. (default: normal).
- `--max-backtracks`: Maximum number of backtracks for pathfinding (default: 5).
- `--output-gpx`: Export the selected route to route.gpx (default: False).
- `--output-html`: Export the selected route to a HTML file (default: False).
- `--help`: View the help page.

## Modification
By default, the program will use the default heuristics, there is the option to create custom heuristics by altering the content in `config.py`. 

Example usage of an 'accessible' route: 

`python main.py --start 53.36486137451511,-1.8160056925378616 --end 53.34344386440596,-1.778107050662822 --distance 10 --tolerance 1 --peak-fitness ignore --fitness-mode custom --normal-nodes top10 --mode normal`

## Server Mode (optional)
A server configuration can be set up by using flask and passing through `server = True` during graph initialisation.

Currently the server mode setup is configured to use 3rd party apis.
Alternatively, a local mode can be setup, which will require a `osm.pbf` file to be download along with topography data, which is available from https://opentopography.org/.

*No wrapper or orchestation layer is provided, implemented one would be advised*

## Citations
Boeing, G. (2024). Modeling and Analyzing Urban Networks and Amenities with OSMnx. Working paper. https://geoffboeing.com/publications/osmnx-paper/

Japan Aerospace Exploration Agency (2021). ALOS World 3D 30 meter DEM. V3.2, Jan 2021. Distributed by OpenTopography. https://doi.org/10.5069/G94M92HB. Accessed: 2025-04-15