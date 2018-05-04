import json

# read in business data
business_files = 'dataset/business.json'
business_data = []
with open (business_files, 'r') as f:
    for line in f:
        business_data.append(json.loads(line))

# extract category information
category = []
for item in business_data:
    category.append(item["categories"])

# extract business with selected category
farmersMarket = []
for item in business_data:
    if item["categories"].count("Farmers Market") != 0:
        print(item)
        
key = ['']
