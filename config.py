## These are example heuristics. Current ones are copies from the 'accessible' heuristics
def custom_highway_score(highway):
    if isinstance(highway, list):
        total = 0
        for h in highway:
            total += custom_highway_score(h)
        return total / len(highway)
    if highway in ["pedestrian", "track", "bridleway", "cycleway", "service"]:
        return 0.25
    elif highway in ["residential", "living_street", "unclassified", "road"]:
        return 1
    elif highway in ["tertiary", "tertiary_link", "secondary", "secondary_link", "primary", "primary_link"]:
        return 1
    elif highway in ["trunk", "trunk_link", "motorway", "motorway_link", "steps", "footway", "path"]:
        return 3
    else:
        return 5

def custom_surface_score(surface):
    if isinstance(surface, list):
        total = 0
        for h in surface:
            total += custom_surface_score(h)
        return total / len(surface)
    if surface in ["paved", "asphalt", "chipseal", "concrete", "concrete:plates", "paving_stones", "bricks", "wood",
                   "rubber", "tiles", "fibre_reinforced_polymer_grate"]:
        return 0.25
    elif surface in ["concrete:lanes", "paving_stones:lanes", "grass_paver", "sett", "metal", "metal_grid"]:
        return 1
    elif surface in ["unhewn_cobblestone", "cobblestone", "compacted"]:
        return 2
    elif surface in ["stepping_stones", "fine_gravel", "gravel", "shells", "pebblestone", "ground", "dirt", "earth",
                     "grass", "mud", "sand", "woodchips", "grit", "salt", "wood"]:
        return 3
    else:
        return 4

def custom_fitness(graph, n1, n2, data):
    return data["length"] * graph.nodes[n2]["fitness"] * data["highway_score"]

