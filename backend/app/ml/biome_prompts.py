"""
biome_prompts.py — hand-written prompts mapped to each of the 8 k-means clusters.

Each cluster index corresponds to a biome label inferred from its centroid
(see data/processed/clusters.csv + cluster.py output). During training, every
prompt is paired with its cluster's centroid (6 feature values) as the target.

Cluster centroids (original units), for reference:
  id  elev_mean  elev_std  slope  bio1(°C)  bio4(season*100)  bio12(mm)   label
   0      483        75      3.7    3.8        559             710        boreal_plains
   1     2575        12      0.4  -20.7        872             800        polar_ice
   2     2665       448     25.1    7.5        407            1013        alpine
   3      816       239     14.9   13.0        281            1946        temperate_rainforest
   4      148        23      3.5   25.8         54            2430        tropical_rainforest
   5     4800       150      9.3   -0.7        663             291        high_mountain
   6     1093        45      2.5   22.4        108             796        savanna
   7      581        24      2.0   23.6        725              59        hot_desert
"""
from __future__ import annotations

CLUSTER_TO_BIOME: dict[int, str] = {
    0: "boreal_plains",
    1: "polar_ice",
    2: "alpine",
    3: "temperate_rainforest",
    4: "tropical_rainforest",
    5: "high_mountain",
    6: "savanna",
    7: "hot_desert",
}

BIOME_PROMPTS: dict[str, list[str]] = {
    "boreal_plains": [
        "cold northern plains with scattered pines",
        "siberian taiga under grey skies",
        "boreal forest stretching to the horizon",
        "flat cool lowland with sparse conifers",
        "subarctic tundra meadow",
        "canadian shield wilderness",
        "quiet frozen lakes and birch trees",
        "long dark winter forest",
        "windswept northern moorland",
        "cool rolling hills of the north",
        "scandinavian lowland forest",
        "permafrost edge with hardy shrubs",
        "chilly flatland dotted with spruce",
        "grey skies over a frozen plain",
        "alaskan interior lowlands",
        "pine forest with a thin layer of snow",
        "northern peat bog in autumn",
        "vast cold plain with little elevation",
        "tundra meets taiga",
        "russia's endless northern forest",
    ],
    "polar_ice": [
        "antarctic ice sheet",
        "frozen polar desert",
        "greenland glacier plateau",
        "endless white ice cap",
        "arctic wasteland of ice",
        "flat frozen expanse at the pole",
        "ice shelf under a pale sun",
        "glacial plateau with no vegetation",
        "subzero polar tundra",
        "blinding white polar ice",
        "frozen continent of snow",
        "high-altitude ice desert",
        "permanent ice sheet",
        "glacier surface stretching for miles",
        "polar night over frozen ground",
        "ice dome of the far north",
        "cold barren polar expanse",
        "snow-covered glacial highland",
        "icy polar flat under blizzard",
        "kilometer-thick ice cap",
    ],
    "alpine": [
        "rugged alpine peaks",
        "snowy mountain valley in the alps",
        "steep glaciated mountains",
        "rocky high-altitude forest",
        "alpine meadow below jagged peaks",
        "dolomite cliffs and pine forest",
        "cool misty mountain range",
        "switzerland in early summer",
        "high alpine ridgeline",
        "craggy peaks with melting glaciers",
        "mountainous region with sharp relief",
        "alpine tundra above the treeline",
        "rocky mountain forest with streams",
        "steep forested slopes with snow",
        "cold mountain wilderness",
        "scree slopes and alpine lakes",
        "rugged range of snow-capped mountains",
        "high cold valleys with pine",
        "rocky mountains national park",
        "cascade range in spring",
    ],
    "temperate_rainforest": [
        "pacific northwest rainforest",
        "misty temperate woodland with ferns",
        "lush mossy forest on steep hills",
        "olympic peninsula valley",
        "rainy coastal mountains with cedars",
        "chilean valdivian forest",
        "dense green forest with heavy rainfall",
        "foggy mountains covered in trees",
        "rugged hills with dripping moss",
        "british columbia coastal range",
        "wet temperate jungle",
        "damp evergreen mountain forest",
        "new zealand south island forest",
        "steep hills soaked in rain",
        "cool cloud forest",
        "japanese cedar mountains in mist",
        "tree-covered foothills with rivers",
        "rain-drenched temperate slopes",
        "mountainous rainforest under grey sky",
        "emerald forested hills",
    ],
    "tropical_rainforest": [
        "amazon rainforest canopy",
        "steamy equatorial jungle",
        "dense tropical forest with rivers",
        "congo basin lowland jungle",
        "humid flat jungle with vines",
        "borneo rainforest",
        "hot wet lowland with tall trees",
        "tropical flat forest near the equator",
        "monsoon jungle in the lowlands",
        "sweltering green jungle",
        "dense tropical canopy",
        "equatorial swamp forest",
        "papua new guinea lowland jungle",
        "amazon floodplain",
        "lush hot humid jungle",
        "rainy tropical basin",
        "thick jungle with no seasons",
        "year-round warm rainforest",
        "southeast asian lowland jungle",
        "flat equatorial wilderness",
    ],
    "high_mountain": [
        "himalayan peaks above the clouds",
        "extreme high altitude mountain range",
        "bare rocky ridges near everest",
        "andean altiplano peaks",
        "tibetan plateau high country",
        "karakoram range",
        "frigid high mountain desert",
        "snowless high rocky summits",
        "rocky terrain above 4000 meters",
        "thin air mountain landscape",
        "dry cold high mountains",
        "pamir mountains in winter",
        "peaks too high for trees",
        "jagged high-altitude rock",
        "himalayan valley with glacial streams",
        "cold arid mountain plateau",
        "bolivian highland peaks",
        "roof of the world",
        "extreme elevation barren mountains",
        "high altitude snow and rock",
    ],
    "savanna": [
        "african savanna grassland",
        "serengeti plain with acacia trees",
        "warm dry grassland with scattered trees",
        "tropical savanna in the dry season",
        "kenyan highland plains",
        "rolling yellow grasslands",
        "flat semi-arid plateau with shrubs",
        "sub-saharan open woodland",
        "cerrado in brazil",
        "dry tropical plateau",
        "warm plains with baobab trees",
        "elevated savanna in east africa",
        "grassy highland in the tropics",
        "open plain dotted with small trees",
        "dry warm rolling grassland",
        "australian outback savanna",
        "warm elevated flatland",
        "tanzania highland plain",
        "llanos of venezuela",
        "savanna before the monsoon",
    ],
    "hot_desert": [
        "sahara desert sand dunes",
        "scorching sandy wasteland",
        "endless sand desert",
        "arabian desert under harsh sun",
        "flat hot desert with no vegetation",
        "sonoran desert flats",
        "dry cracked desert floor",
        "desert with towering dunes",
        "atacama desert plateau",
        "bone-dry flat wilderness",
        "gobi desert in summer",
        "red sand desert at noon",
        "barren hot plain",
        "desolate sandy expanse",
        "namib desert sea of sand",
        "hot rocky desert flats",
        "kalahari sand country",
        "wind-carved desert plain",
        "desert of shifting dunes",
        "mojave desert basin",
    ],
}


def iter_training_pairs() -> list[tuple[str, int]]:
    """Yield (prompt, cluster_id) pairs for training."""
    biome_to_cluster = {v: k for k, v in CLUSTER_TO_BIOME.items()}
    pairs: list[tuple[str, int]] = []
    for biome, prompts in BIOME_PROMPTS.items():
        cluster_id = biome_to_cluster[biome]
        for p in prompts:
            pairs.append((p, cluster_id))
    return pairs


if __name__ == "__main__":
    pairs = iter_training_pairs()
    print(f"Total prompts: {len(pairs)}")
    for biome, prompts in BIOME_PROMPTS.items():
        print(f"  {biome}: {len(prompts)}")
