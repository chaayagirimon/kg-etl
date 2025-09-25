import os
import shutil
import json

# airbnb_listings_og_path = "/Users/chaayagirimon/Shelf/thesis/datasets/airbnb/listings"
# airbnb_reviews_og_path = "/Users/chaayagirimon/Shelf/thesis/datasets/airbnb/reviews"

# airbnb_destination_path = "/Users/chaayagirimon/Shelf/thesis/datasets/etl_full_reviewsdb_v3/data/airbnb/raw"

# for file in os.listdir(airbnb_listings_og_path):
#     if file.endswith(".csv"):
#         shutil.copy(os.path.join(airbnb_listings_og_path, file), os.path.join(airbnb_destination_path, file))

# for file in os.listdir(airbnb_reviews_og_path):
#     if file.endswith(".csv"):
#         shutil.copy(os.path.join(airbnb_reviews_og_path, file), os.path.join(airbnb_destination_path, file))

alias_json = "/Users/chaayagirimon/Shelf/thesis/datasets/kg_etl/diagnostics/wv_alias_suggestions.json"
config_json = "/Users/chaayagirimon/Shelf/thesis/datasets/kg_etl/cities_config.generated.json"
og_config = "/Users/chaayagirimon/Shelf/thesis/datasets/kg_etl/cities_config.generated.json 00-33-11-153.json"

with open(alias_json, "r") as f:
    alias_data = json.load(f)

with open(config_json, "r") as f:
    cfg = json.load(f)

with open(og_config, 'r') as f: 
    og_cfg  = json.load(f)

new_cfg_list = []

city_list = [city for city in alias_data]
# print(city_list, len(city_list))

for city_slug, aliases in alias_data.items():
    # print(city_slug)
    flag = 0
    for city in og_cfg["cities"]:
        # import ipdb; ipdb.set_trace()
        if city not in new_cfg_list:
            if city["slug"] not in city_list: 
                city["aliases"] = []
                flag = 1
            elif city["slug"] == city_slug:
                city["aliases"] = aliases 
                flag = 1
            new_cfg_list.append(city)
            # break
        


final_cfg = {"cities": new_cfg_list}

with open("try_config.json", "w") as f:
    json.dump(final_cfg, f, indent=2, ensure_ascii=False)