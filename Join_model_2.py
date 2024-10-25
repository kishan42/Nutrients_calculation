import os
import google.generativeai as genai
import streamlit as st
import json
import pymongo
import urllib
import urllib.parse
import json
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
import nltk
nltk.download('punkt_tab')
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Model 1
genai.configure(api_key="AIzaSyC-0EixxxNZWqRp7wYsEOYrMFsI81iPqHk")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
  system_instruction="input : Shrikhand\noutput :\n{\n  \"Dish Name\": \"Shrikhand\",\n  \"Dish Description\": \"A creamy, sweet Indian dessert made with hung curd, cardamom, saffron, and sugar.\",\n  \"Cuisine Category\": \"Asia - Indian - North Indian\",\n  \"Food Category\": \"Vegetarian\",\n  \"Timing Category\": [\n    \"Dessert\"\n  ],\n  \"Meal Category\": [\n    \"Dessert\"\n  ],\n  \"Cooking Method\": [\n    \"Mixing\"\n  ],\n  \"Cooking Time\": \"30 minutes\",\n  \"Cooking Temperature\": \"N/A\",\n  \"Ingredients\": [\n    \"Hung Curd - 150g\",\n    \"Sugar - 30g\",\n    \"Cardamom Powder - 1.9g\",\n    \"Saffron - 0.1g\",\n    \"Pistachios - 8g\",\n    \"Milk - 10ml\"\n  ],\n  \"Allergic Content\": {\n    \"Milk and Dairy Products\": [\n      \"Hung Curd\"\n    ]\n  },\n  \"Serving Size\": \"200g\",\n  \"Cooking Steps\": [\n    \"1.  Strain plain yogurt through a cheesecloth or fine-mesh sieve for 3-4 hours to make hung curd.\",\n    \"2.  In a bowl, whisk together the hung curd, sugar, and cardamom powder until smooth.\",\n    \"3.  Soak the saffron strands in a few drops of warm milk for about 10 minutes to extract color and flavor.\",\n    \"4.  Add the saffron milk to the curd mixture and whisk well.\",\n    \"5.  Refrigerate for at least 2 hours to allow the flavors to meld.\",\n    \"6.  Garnish with chopped pistachios before serving.\"\n  ]\n}\n\nrule for data generation :\nhere total serving size must be 200g or 200ml based on dish if it solid then 200g and liquid then 200ml , based on this  serving size 200, total sum of weight of all ingredient must be 200 so write ingredients weight with proportional to 200, example that here see that 150 + 30 + 1.9 + 0.1 + 8 + 10 = 200 and  same as serving size write the quntity in g or ml based material type so unit of volume must be in g and ml quantity , not aacept any other unit of volume if it is then convert into g or ml then write based on that. for cooking method use only this standard names from this \"Boiling, Steaming, Microwaving, Baking, Roasting, Grilling, Broiling, Frying, Sautéing, Stir-frying, Pressure Cooking, Slow Cooking\" not allowed any other text in unit. For the timing category, use only these standard names from this Breakfast, Lunch, Snack, Dinner, Brunch, Midnight Snack, not allowed any other text in the unit. For the meal category, use only  from this \"Finger Food, Accompaniments,Appetizer, Soup, Salad, Main Course,Side Dish,Accompaniments,Dessert,Beverage,Drink,Snack\" not allowed any other text in unit. then for cooking temprature must be in °C unit not allowed any other text or unit.\n\ninput : Baida Roti\noutput :\n{\n  \"Dish Name\": \"Baida Roti\",\n  \"Dish Description\": \"A type of flatbread made with eggs, flour, and spices, originating from the Indian subcontinent.\",\n  \"Cuisine Category\": \"Asia - Indian - North Indian\",\n  \"Food Category\": \"Non-vegetarian\",\n  \"Timing Category\": [\n    \"Breakfast\",\n    \"Lunch\",\n    \"Dinner\"\n  ],\n  \"Meal Category\": [\n    \"Bread\",\n    \"Side Dish\"\n  ],\n  \"Cooking Method\": [\n    \"Grilling\"\n  ],\n  \"Cooking Time\": \"15 minutes\",\n  \"Cooking Temperature\": \"100°C\",\n  \"Ingredients\": [\n    \"Eggs - 100g\",\n    \"Wheat Flour - 80g\",\n    \"Milk - 10ml\",\n    \"Butter - 5g\",\n    \"Cumin Seeds - 2g\",\n    \"Salt - 2g\",\n    \"Red Chili Powder - 1g\"\n  ],\n  \"Allergic Content\": {\n    \"Eggs\": [\n      \"Eggs\"\n    ],\n    \"Milk and Dairy Products\": [\n      \"Milk\",\n      \"Butter\"\n    ]\n  },\n  \"Serving Size\": \"200g\",\n  \"Cooking Steps\": [\n    \"1. Crack eggs into a bowl and whisk them well.\",\n    \"2. Add wheat flour, milk, melted butter, cumin seeds, salt, and red chili powder to the whisked eggs. Mix well to form a smooth dough.\",\n    \"3. Heat a flat griddle or tava over medium heat.\",\n    \"4. Divide the dough into equal portions and roll each portion into a thin circle.\",\n    \"5. Place the rolled roti on the hot griddle and cook for 2-3 minutes per side, or until golden brown and cooked through.\",\n    \"6. Serve hot with butter, chutney, or your favorite curry.\"\n  ]\n}\n\ninput : Kaju Curry\noutput :\n{\n  \"Dish Name\": \"Kaju Curry\",\n  \"Dish Description\": \"A creamy, sweet, and slightly tangy curry made with cashews, coconut milk, and aromatic spices.\",\n  \"Cuisine Category\": \"Asia - Indian - South Indian\",\n  \"Food Category\": \"Vegetarian\",\n  \"Timing Category\": [\n    \"Lunch\",\n    \"Dinner\"\n  ],\n  \"Meal Category\": [\n    \"Main Course\",\n    \"Side Dish\"\n  ],\n  \"Cooking Method\": [\n    \"Sautéing\"\n  ],\n  \"Cooking Time\": \"30 minutes\",\n  \"Cooking Temperature\": \"80°C\",\n  \"Ingredients\": [\n    \"Cashews - 40g\",\n    \"Coconut Milk - 80ml\",\n    \"Onions - 20g\",\n    \"Tomatoes - 25g\",\n    \"Ginger-Garlic Paste - 5g\",\n    \"Green Chilies - 2g\",\n    \"Turmeric Powder - 1g\",\n    \"Red Chili Powder - 2g\",\n    \"Cumin Powder - 1g\",\n    \"Coriander Powder - 2g\",\n    \"Garam Masala - 1g\",\n    \"Salt - 2g\",\n    \"Oil - 10ml\"\n  ],\n  \"Allergic Content\": {\n    \"Tree Nuts\": [\n      \"Cashews\"\n    ]\n  },\n  \"Serving Size\": \"200g\",\n  \"Cooking Steps\": [\n    \"1. Soak cashews in warm water for 15 minutes. Drain and grind them into a smooth paste.\",\n    \"2. Heat oil in a pan. Add chopped onions and green chilies. Sauté until onions turn translucent.\",\n    \"3. Add ginger-garlic paste, turmeric powder, red chili powder, cumin powder, and coriander powder. Sauté for a minute until fragrant.\",\n    \"4. Add chopped tomatoes and cook until they soften.\",\n    \"5. Add the cashew paste, coconut milk, garam masala, and salt. Stir well and simmer for 10-15 minutes, stirring occasionally.\",\n    \"6. Garnish with fresh coriander leaves (optional) and serve hot with rice or roti.\"\n  ]\n}\n\ninput : \noutput :\n{\n  \"Dish Name\": \"Chicken Parmesan\",\n  \"Dish Description\": \"Breaded chicken cutlets topped with marinara sauce and melted mozzarella cheese.\",\n  \"Cuisine Category\": \"Europe - Italian - Italian-American\",\n  \"Food Category\": \"Non-vegetarian\",\n  \"Timing Category\": [\n    \"Dinner\",\n    \"Lunch\"\n  ],\n  \"Meal Category\": [\n    \"Main Course\"\n  ],\n  \"Cooking Method\": [\n    \"Baking\",\n    \"Frying\"\n  ],\n  \"Cooking Time\": \"30 minutes\",\n  \"Cooking Temperature\": \"180°C\",\n  \"Ingredients\": [\n    \"Chicken Breast - 80g\",\n    \"Bread Crumbs - 20g\",\n    \"Parmesan Cheese - 10g\",\n    \"Egg - 50g\",\n    \"Marinara Sauce - 20ml\",\n    \"Mozzarella Cheese - 10g\",\n    \"Olive Oil - 8.5ml\",\n    \"Salt - 1g\",\n    \"Pepper - 0.5g\"\n  ],\n  \"Allergic Content\": {\n    \"Eggs\": [\n      \"Egg\"\n    ],\n    \"Milk and Dairy Products\": [\n      \"Parmesan Cheese\",\n      \"Mozzarella Cheese\"\n    ]\n  },\n  \"Serving Size\": \"200g\",\n  \"Cooking Steps\": [\n    \"1. Preheat the oven to 180°C. \",\n    \"2. Pound the chicken breast to create thin cutlets. \",\n    \"3. In a shallow dish, whisk the egg with a pinch of salt and pepper. \",\n    \"4. In another shallow dish, combine breadcrumbs and grated Parmesan cheese.\",\n    \"5. Dip each chicken cutlet into the egg mixture, then dredge in the breadcrumb mixture, ensuring it is fully coated. \",\n    \"6. Heat olive oil in a large skillet over medium heat. \",\n    \"7. Fry the breaded chicken cutlets for 3-4 minutes per side, or until golden brown and cooked through.\",\n    \"8. Place the cooked chicken cutlets on a baking sheet. \",\n    \"9. Top each cutlet with marinara sauce and shredded mozzarella cheese. \",\n    \"10. Bake in the preheated oven for 5-7 minutes, or until the cheese is melted and bubbly.\",\n    \"11. Serve hot with a side of pasta or vegetables.\"\n  ]\n}",
)


# Model 2
# Encode the username and password
username = urllib.parse.quote_plus("Kishan")
password = urllib.parse.quote_plus("KishankFitshield")  # Encode special characters
# MongoDB connection
uri = f"mongodb://{username}:{password}@ec2-13-233-104-209.ap-south-1.compute.amazonaws.com:27017/?authMechanism=SCRAM-SHA-256&authSource=Fitshield"
client = pymongo.MongoClient(uri)
db = client["Fitshield"]  # Database name
dish_collection = db["Dishdetails"]  # Collection for dish details
converse = db["Unitconversion"]  # Collection for density values
nutrients_collection = db['Nutrients'] # collection for nutrients value
retention_collection = db["NutrientRetention"]  # Collection for nutrient retention values

api_key = "c2468019-302c-411e-bf36-c802d54d7a87"
unit_index_name ="hybrid-search-langchain-unitconversion"
ingredient_index_name="hybrid-search-langchain-pincone"

## intialize pinecone client
pc = Pinecone(api_key=api_key)
unit_index=pc.Index(unit_index_name)
ingredient_index = pc.Index(ingredient_index_name)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

ingredient_bm25_encoder = BM25Encoder().load("ingredients_bm25_values.json")
unit_bm25_encoder = BM25Encoder().load("unit_bm25_values.json")
unit_retriever = PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=unit_bm25_encoder,index=unit_index)
ingredient_retriever = PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=ingredient_bm25_encoder,index=ingredient_index)

# Function to convert ml to grams based on density
def convert_ml_to_g(ingredient_name, quantity_ml,retriever):
    # Fetch the density value from the database
    simentic_search = retriever.invoke(ingredient_name)[0].page_content

    density_data = converse.find_one({"ingredient": simentic_search})
    #print(density_data,ingredient_name, quantity_ml)
    if density_data and 'density' in density_data:
        density = density_data['density']  # density is in g/mL
        # Convert ml to grams using the formula: grams = ml * density
        quantity_g = quantity_ml * density
        #print(quantity_g,quantity_ml,density)
        return quantity_g  # Round to 2 decimal places
    else:
        return None  # If no density data found

# Function to calculate nutrient values for an ingredient
def calculate_nutrients(ingredient_name, quantity_g, retriever):
    # Fetch the nutrient data from the database

    simentic_search = retriever.invoke(ingredient_name)[0].page_content
    
    #print(ingredient_name,simentic_search)

    nutrient_data = nutrients_collection.find_one({"Food name": simentic_search})
    if not nutrient_data:
        return None  # If no nutrient data is found for the ingredient
    
    nutrients_result = {}
    
    # Iterate over all nutrient fields in the nutrient data
    for nutrient, value in nutrient_data.items():
        
        if nutrient == "_id" or nutrient == "Food name" or nutrient == "Foodcode":
            continue  # Skip non-nutrient fields
        
        if value:  # Check if the string is not empty
            try:
                base_value = float(value)  # Try to convert to float
            except ValueError:
                base_value = 0  # Set to 0 if conversion fails
        else:
            base_value = 0  # Set to 0 if the value is empty or invali
        
        
        # Calculate the nutrient value for the given quantity in grams
        calculated_value = (base_value * quantity_g) / 100
        # Store the calculated nutrient value
        nutrients_result[nutrient] = round(calculated_value, 2)
    
    return nutrients_result   

# Function to adjust total nutrient values based on cooking method retention
def apply_cooking_retention(total_nutrients, cooking_method):
    # Fetch the retention data from the database for the given cooking method
    retention_data = retention_collection.find_one({"Cooking Method": cooking_method})
    
    if retention_data:
        for nutrient, retention_factor in retention_data.items():
            if nutrient in ["_id", "Cooking Method"]:
                continue  # Skip non-nutrient fields
            
            # If the nutrient exists in total_nutrients, apply the retention factor
            if nutrient in total_nutrients:
                #print(nutrient,total_nutrients,total_nutrients[nutrient],retention_factor)
                total_nutrients[nutrient] = round(total_nutrients[nutrient] * retention_factor, 2)
                #print(total_nutrients[nutrient])
    
    return total_nutrients
    


# Streamlit app interface
st.title("Dish Information Generator")

# Input: dish name from user
dish_name = st.text_input("Please enter the name of the dish:")


if dish_name:
    chat_session = model.start_chat(
            history=[
    {
      "role": "user",
      "parts": [
        "you are expert and knowledge about recipe of world and you have health concise regarding food and its ingredients and Allergic Content. you have to knowledge about this all.\nso i will provide any dish name, for example \"Paneer Taash Kabab\"\nand based on this dish name generate dish information for example,\n{\n\"Dish Name\": \"Paneer Taash Kabab\",\n\"Dish Description\": \"A feast for vegetarians, layers of paneer slices filled with chutney, vegetables, cheese, and seasoning, topped with more cheese, cream, and butter, and baked.\",\n\"Cuisine Category\": \"Asia - Indian - North Indian\",\n\"Timing Category\": [\n\"Snack\",\n\"Dinner\"\n],\n\"Meal Category\": [\n\"Appetizer\",\n\"Main Course\"\n],\n\"Cooking Method\": [\n\"Baking\"\n],\n\"Cooking Time\": \"45 minutes\",\n\"Cooking Temperature\": \"180°C\",\n\"Ingredients\": [\n\"Paneer - 70g\",\n\"Cheese Slices - 15g\",\n\"Onions - 10g\",\n\"Tomatoes - 10g\",\n\"Mint Chutney - 15g\",\n\"Grated Cheese - 10g\",\n\"Fresh Cream - 5g\",\n\"Hung Yogurt - 20g\",\n\"Red Chili Powder - 3g\",\n\"Coriander Powder - 2g\",\n\"Cumin Powder - 2g\",\n\"Ginger-Garlic Paste - 5g\",\n\"Garam Masala - 2g\",\n\"Mustard Oil - 5ml\",\n\"Chaat Masala - 2g\",\n\"Salt - 2.79g\",\n\"Black Pepper - 2g\"\n],\n\"Allergic Content\": {\n\"Milk and Dairy Products\": [\n\"Paneer\",\n\"Cream\",\n\"Cheese\"\n]\n},\n\"Serving Size\": \"200g\",\n\"Cooking Steps\": [\n\"1. Preheat the oven to 180°C.\",\n\"2. Slice the paneer into thin pieces and set aside.\",\n\"3. Mix hung yogurt with red chili powder, coriander powder, cumin powder, ginger-garlic paste, garam masala, mustard oil, salt, and black pepper to make the marinade.\",\n\"4. Spread mint chutney on half of the paneer slices and top with thin slices of onions and tomatoes. Add cheese slices on top of the vegetables.\",\n\"5. Cover with the remaining paneer slices to create layers. Brush the tops with the prepared marinade.\",\n\"6. Place the layered paneer in a baking dish and top with grated cheese, fresh cream, and a sprinkle of chaat masala.\",\n\"7. Bake in the preheated oven for 20-25 minutes or until the paneer is golden and the cheese has melted.\",\n\"8. Remove from the oven and serve hot.\"\n]\n},\nwhere use below validation rule and format\nDish Description :\nwhere\nCharacter Limit: 20 to 200 characters (A crisp description is key).\nValidation Rule: The description should provide an appetizing overview,\nhighlighting the dish’s main ingredients, flavor profile, and style of preparation.\nAvoid excessive use of special characters unless they are part of the description\n(e.g., apostrophes or hyphens).\nExample: “A spicy stir-fried paneer dish with bell peppers and onions in a tangy\nsoy sauce.”\nCuisine category:\nwhere, format should strictly follow the structure [Continental Region] -\n[Specific Cuisine] - [Subcategory]\nContinental Region:\no Validation Criteria:\nMust be one of the predefined regions: Asia, Europe, Africa, North\nAmerica, South America, Middle East, Oceania\nExample Validation Rules:\nAcceptable: Asia, europe, MIDDLE EAST.\nReject: Asis (typo), Eurasia (invalid region).\nSpecific Cuisine:\no Validation Criteria:\nMust match a predefined list of cuisines associated with the selected\ncontinental region.\nExample Validation Rules:\nAcceptable for Asia: Indian, Chinese, Thai.\nReject: Indian under Europe (mismatch), Europian (invalid spelling).\nSubcategory:\no Validation Criteria:\nMust match a predefined list of subcategories under the selected cuisine.\nShould be case-insensitive.\no Example Validation Rules:\nAcceptable for Indian: North Indian, South Indian, Indo-Chinese.\nReject: Szechuan (incorrect subcategory for Indian), Nort India (Typo).\nFormat Validation:\nThe format should strictly follow the structure [Continental Region] -\n[Specific Cuisine] - [Subcategory].\nHyphens should separate each level without extra spaces.\no Example Validation Rules:\nAcceptable: Asia - Indian - North Indian, Europe - Italian - Tuscan.\nReject: Asia-Indian-NorthIndian (missing spaces), Europe--Italian-Tuscan\n(extra hyphens), ASIA - INDIAN (missing subcategory).\nTiming category: Breakfast/ lunch/snack/dinner (Select applicable one or many)\n▪ Meal Timing Categories:\no Breakfast\no Lunch\no Snack\no Dinner\no Brunch (optional)\no Midnight Snack (optional)\nCriteria: Ensure the input is from a predefined list of timing categories.\no Acceptable: \"Breakfast\", \"Lunch\", \"Snack\", \"Dinner\", \"Brunch\", \"Midnight\nSnack\".\no Reject: Variants like \"brunch time\", \"dinner meal\", or misspellings like \"Dinner\".\ntiming category value is write based on the priority wise like if any dish is eat at dinner time then first write Dinner then also if it eat at Snack so write it else don't write.\nCooking method\nInput Validation Steps:\n▪ Predefined List Matching:\no Validate that the cooking method provided is within an approved list:\no Example: [\"Boiling\", \"Steaming\", \"Microwaving\", \"Baking\", \"Grilling\", \"Frying\",\n\"Sautéing\", \"Pressure Cooking\", \"Slow Cooking\"]\nCooking time\nInput Format for Cooking Time:\n▪ Expected Data: A numerical value with time units (e.g., minutes or hours).\n▪ Standard Input Format: \"X minutes\" or \"Y hours\"\n▪ Examples:\no \"30 minutes\" or \"1 hour 15 minutes\"\no Valid: \"45 mins\", \"1 hour\"\no Invalid: \"45m\", \"1 h\"\n▪ Validate Input Range:\nEnsure cooking times are within a reasonable range for the dish.\no Typical Ranges:\nQuick Cook (under 30 minutes)\nMedium Cook (30-60 minutes)\nLong Cook (1-3 hours)\nExtended Cook (Over 3 hours)\n▪ Check for Missing Units:\no If the input only contains a number without a time unit, prompt for correction.\no Example: \"45\" should ask, “Is this in minutes or hours?”\n▪ Handling Edge Cases:\no Approximate Times: Inputs like “around 20 minutes” should be captured as 20\nminutes.\no Multiple Times: If multiple time ranges are mentioned (e.g., “30-40 minutes”),\ntake the average or prompt for a specific choice.\no Non-Numeric Descriptions: If inputs like “quick cook” are used, map them to a\ntypical time range (e.g., “under 30 minutes”).\n▪ Final output = \"Dish Name\": \"Paneer Chili\"- \"Cooking Time\": \"30 minutes\"\nCooking temperature :\nExpected Data Type: Numeric (float or integer)\n▪ Unit of Measurement: Degrees Celsius (°C)\nValid Range: Typically, 50°C to 500°C\n▪ Input Example: \"180°C\"\nIngredients :\nwrite list for Ingredient based on dish and write on priority wise like any ingredients are most usable in dish then write first then others accordingly\nuse format for ingredients , \"Ingredient name - weights in g or ml\"\nwrite ingredients weights according with serving size is 200g or 200ml because serving size is sum of weights of all ingredients\nfor example,\n\"Ingredients\": [\n\"Frozen Blueberries - 70g\",\n\"Pomegranate Juice - 30ml\",\n\"Plain Yogurt - 70g\",\n\"Honey or Maple Syrup - 30ml\" ]\nAllergic content :\nData Input\no Description: Free text description of the dish.\no Ingredients: List of ingredients used in the dish.\n▪ Preprocessing\no Text Normalization: Convert text to lowercase, remove special characters, and\nperform stemming/lemmatization.\no Tokenization: Split text into tokens (words or phrases).\n▪ Allergen Detection\no Keywords Extraction: Identify key allergen-related terms from both the\ndescription and ingredients.\no Pattern Matching: Use pattern matching techniques to find allergens based on\npredefined lists.\n▪ Validation Logic\no Allergen Mapping: Map detected keywords to allergen categories based on a\npredefined allergen list.\no Contextual Analysis: Ensure that detected allergens are relevant in the context\n(e.g., avoid false positives from irrelevant terms)\nuse standard name of Allergic content categories like,\nCereals Containing Gluten\nCrustaceans\nEggs\nFish\nPeanuts\nSoybeans\nMilk and Dairy Products\nTree Nuts\nSulphites\nSesame Seeds\nMustard\nMolluscs\nLupin\nGluten-Free Ingredients\nCelery\nSoy\nAlcohol\nTapioca\nServing size:\nserving size is must be calculate that total sum of weights of all ingredients which is g and ml\n▪ Format Validation\no Requirement: Ensure the serving size is entered in a standard, consistent\nformat.\no Examples: Grams (g), milliliters (ml).\n▪ Range Validation\no Requirement: Verify that the serving size is 200g or 200ml.\no Examples:\nFor solids: 200g\nFor liquids: 200ml\no Validation:\nCompare the serving size with 200ml or 200g for different food types.\nFlag any values that fall outside these ranges for review.\nCooking steps :\nwrite cooking steps based on the dish name\nfor example,\n\"Cooking Steps\": [\n\"1. Combine all ingredients in a blender.\",\n\"2. Blend on high speed until the smoothie is smooth and creamy.\",\n\"3. Adjust the sweetness with honey or maple syrup to your preference.\",\n\"4. Pour into a glass and serve immediately.\"\n]\nplease must be refer and strictly follow this validation rules and format then after generate output",
      ],
    },
    {
      "role": "model",
      "parts": [
        "```json\n{\n\"Dish Name\": \"Paneer Taash Kabab\",\n\"Dish Description\": \"Layers of paneer slices filled with chutney, vegetables, and cheese, then baked with cheese, cream, and butter for a rich, savory treat.\",\n\"Cuisine Category\": \"Asia - Indian - North Indian\",\n\"Timing Category\": [\n\"Dinner\",\n\"Snack\"\n],\n\"Meal Category\": [\n\"Appetizer\",\n\"Main Course\"\n],\n\"Cooking Method\": [\n\"Baking\"\n],\n\"Cooking Time\": \"45 minutes\",\n\"Cooking Temperature\": \"180°C\",\n\"Ingredients\": [\n\"Paneer - 70g\",\n\"Cheese Slices - 15g\",\n\"Onions - 10g\",\n\"Tomatoes - 10g\",\n\"Mint Chutney - 15g\",\n\"Grated Cheese - 10g\",\n\"Fresh Cream - 5g\",\n\"Hung Yogurt - 20g\",\n\"Red Chili Powder - 3g\",\n\"Coriander Powder - 2g\",\n\"Cumin Powder - 2g\",\n\"Ginger-Garlic Paste - 5g\",\n\"Garam Masala - 2g\",\n\"Mustard Oil - 5ml\",\n\"Chaat Masala - 2g\",\n\"Salt - 2.79g\",\n\"Black Pepper - 2g\"\n],\n\"Allergic Content\": {\n\"Milk and Dairy Products\": [\n\"Paneer\",\n\"Cream\",\n\"Cheese\"\n]\n},\n\"Serving Size\": \"200g\",\n\"Cooking Steps\": [\n\"1. Preheat the oven to 180°C.\",\n\"2. Slice paneer into thin pieces.\",\n\"3. Mix hung yogurt with spices: red chili powder, coriander powder, cumin powder, ginger-garlic paste, garam masala, mustard oil, salt, and black pepper.\",\n\"4. Spread mint chutney on half the paneer slices, top with onion and tomato slices, then add cheese slices.\",\n\"5. Cover with remaining paneer slices to create layers. Brush marinade on top.\",\n\"6. Place in a baking dish. Top with grated cheese, fresh cream, and chaat masala.\",\n\"7. Bake for 20-25 minutes until paneer is golden and cheese is melted.\",\n\"8. Serve hot.\"\n]\n}\n``` \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Caramel Latte",
      ],
    },
    {
      "role": "model",
      "parts": [
        "```json\n{\n\"Dish Name\": \"Caramel Latte\",\n\"Dish Description\": \"A creamy, sweet coffee drink with caramel flavor.\",\n\"Cuisine Category\": \"North America - American - Coffee Drinks\",\n\"Timing Category\": [\n\"Snack\",\n\"Breakfast\",\n\"Lunch\"\n],\n\"Meal Category\": [\n\"Beverage\"\n],\n\"Cooking Method\": [\n\"Boiling\",\n\"Steaming\"\n],\n\"Cooking Time\": \"5 minutes\",\n\"Cooking Temperature\": \"N/A\",\n\"Ingredients\": [\n\"Espresso - 60ml\",\n\"Milk - 150ml\",\n\"Caramel Syrup - 15ml\",\n\"Whipped Cream - 30ml\",\n\"Caramel Sauce - 10ml\"\n],\n\"Allergic Content\": {\n\"Milk and Dairy Products\": [\n\"Milk\",\n\"Whipped Cream\"\n]\n},\n\"Serving Size\": \"200ml\",\n\"Cooking Steps\": [\n\"1. Brew a shot of espresso.\",\n\"2. Heat milk in a saucepan or steam it using a milk frother.\",\n\"3. Pour the espresso into a mug.\",\n\"4. Add caramel syrup to the espresso.\",\n\"5. Pour the heated milk over the espresso and syrup.\",\n\"6. Top with whipped cream and drizzle caramel sauce.\"\n]\n}\n``` \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "here total weights of ingredients is more then 200g so please must be all ingredients weights sum is 200g or 200ml so set weights of all ingredients based on sum of 200g or 200ml",
      ],
    },
    {
      "role": "model",
      "parts": [
        "You are absolutely right! My apologies for the oversight.  Here's the corrected JSON output with the ingredient weights adjusted to total 200ml:\n\n```json\n{\n\"Dish Name\": \"Caramel Latte\",\n\"Dish Description\": \"A creamy, sweet coffee drink with caramel flavor.\",\n\"Cuisine Category\": \"North America - American - Coffee Drinks\",\n\"Timing Category\": [\n\"Snack\",\n\"Breakfast\",\n\"Lunch\"\n],\n\"Meal Category\": [\n\"Beverage\"\n],\n\"Cooking Method\": [\n\"Boiling\",\n\"Steaming\"\n],\n\"Cooking Time\": \"5 minutes\",\n\"Cooking Temperature\": \"N/A\",\n\"Ingredients\": [\n\"Espresso - 40ml\",\n\"Milk - 130ml\",\n\"Caramel Syrup - 15ml\",\n\"Whipped Cream - 10ml\",\n\"Caramel Sauce - 5ml\"\n],\n\"Allergic Content\": {\n\"Milk and Dairy Products\": [\n\"Milk\",\n\"Whipped Cream\"\n]\n},\n\"Serving Size\": \"200ml\",\n\"Cooking Steps\": [\n\"1. Brew a shot of espresso.\",\n\"2. Heat milk in a saucepan or steam it using a milk frother.\",\n\"3. Pour the espresso into a mug.\",\n\"4. Add caramel syrup to the espresso.\",\n\"5. Pour the heated milk over the espresso and syrup.\",\n\"6. Top with whipped cream and drizzle caramel sauce.\"\n]\n}\n```\n\nI've adjusted the ingredient quantities to ensure the total serving size is 200ml.  Let me know if you have any other requests! \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "please keep this rule for further conservations",
      ],
    },
    {
      "role": "model",
      "parts": [
        "You got it! I will make sure to keep the ingredient weight total at 200g (for solid ingredients) or 200ml (for liquid ingredients) for every dish you ask about.  Let me know what other recipes you'd like information on. I'm ready to dig into the culinary world!  \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "if total sum of ingredients weights is less than 200g or 200ml so increase ingredients weights according to set sum of ingredients weight is 200g or 200ml becuase its service size is fixed 200g or 200ml for all dishes.",
      ],
    },
    {
      "role": "model",
      "parts": [
        "You are absolutely right! I'm still under development, and I'm learning to handle those edge cases.  You're correct that if the total ingredients are less than 200g/ml, we need to adjust them to reach that target. \n\nLet's revise the \"Fat Free Chocolate Milk Jug\" example:\n\n```json\n{\n\"Dish Name\": \"Fat Free Chocolate Milk Jug\",\n\"Dish Description\": \"A refreshing and low-fat beverage made with milk and chocolate flavoring.\",\n\"Cuisine Category\": \"North America - American - Dairy Drinks\",\n\"Timing Category\": [\n\"Snack\",\n\"Breakfast\",\n\"Lunch\"\n],\n\"Meal Category\": [\n\"Beverage\"\n],\n\"Cooking Method\": [\n\"Mixing\"\n],\n\"Cooking Time\": \"1 minute\",\n\"Cooking Temperature\": \"N/A\",\n\"Ingredients\": [\n\"Fat Free Milk - 170ml\",\n\"Chocolate Syrup - 20ml\",\n\"Cocoa Powder - 10g\"\n],\n\"Allergic Content\": {\n\"Milk and Dairy Products\": [\n\"Fat Free Milk\"\n]\n},\n\"Serving Size\": \"200ml\",\n\"Cooking Steps\": [\n\"1. Pour the fat free milk into a jug or container.\",\n\"2. Add the chocolate syrup and cocoa powder to the milk.\",\n\"3. Stir or shake the jug until the chocolate syrup and cocoa powder are fully mixed and dissolved in the milk.\",\n\"4. Refrigerate until chilled and enjoy!\"\n]\n}\n```\n\nI've increased the amounts of the milk, chocolate syrup, and cocoa powder to reach a total of 200ml.  \n\nI'll keep this in mind for all future dish descriptions!  Thank you for pointing that out -  it helps me get better at providing accurate information. \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "please keep this rule in your mind",
      ],
    },
    {
      "role": "user",
      "parts": [
        "here if i enter wrong name or spelling mistake in input dish name so make sure correct it your side and find details based on that , ",
        "for example, I write dish name : paneer gilli , by mistake so make use your knowledge and understand perfect name , paneer chilli for that and write this in output and give me details based on that\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "You're right! It's important to be able to handle those kinds of user errors. I'm still learning how to do that, but I'll do my best to understand what you mean, even if there are typos or misspellings. \n\nBased on \"paneer gilli\", I understand you're probably referring to \"Paneer Chilli\". Let's see what I can put together:\n\n```json\n{\n\"Dish Name\": \"Paneer Chilli\",\n\"Dish Description\": \"A spicy stir-fried paneer dish with bell peppers and onions in a tangy soy sauce.\",\n\"Cuisine Category\": \"Asia - Indian - North Indian\",\n\"Timing Category\": [\n\"Dinner\",\n\"Lunch\",\n\"Snack\"\n],\n\"Meal Category\": [\n\"Appetizer\",\n\"Main Course\"\n],\n\"Cooking Method\": [\n\"Stir-frying\"\n],\n\"Cooking Time\": \"30 minutes\",\n\"Cooking Temperature\": \"Medium-High Heat\",\n\"Ingredients\": [\n\"Paneer - 80g\",\n\"Bell Peppers - 50g\",\n\"Onions - 30g\",\n\"Soy Sauce - 20ml\",\n\"Tomato Ketchup - 10ml\",\n\"Ginger-Garlic Paste - 5g\",\n\"Green Chilies - 2g\",\n\"Red Chili Powder - 2g\",\n\"Cumin Powder - 1g\",\n\"Coriander Powder - 1g\",\n\"Salt - 2g\",\n\"Oil - 10ml\"\n],\n\"Allergic Content\": {\n\"Soybeans\": [\n\"Soy Sauce\"\n]\n},\n\"Serving Size\": \"200g\",\n\"Cooking Steps\": [\n\"1. Cut paneer into cubes and marinate with ginger-garlic paste, red chili powder, cumin powder, coriander powder, and salt for 15 minutes.\",\n\"2. Heat oil in a wok or pan. Add chopped onions and green chilies. Sauté until onions turn translucent.\",\n\"3. Add bell peppers and cook for a few minutes until slightly softened.\",\n\"4. Add marinated paneer and stir-fry for 5-7 minutes until paneer is golden brown.\",\n\"5. Add soy sauce and tomato ketchup. Stir well and cook for 2-3 minutes.\",\n\"6. Garnish with chopped cilantro (optional) and serve hot with rice or roti.\"\n]\n}\n```\n\nI've provided information based on the corrected dish name \"Paneer Chilli\".  I'll continue to learn and improve my ability to understand variations in dish names! \n\nLet me know if you have any other dishes in mind. \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "tummer bhicken",
      ],
    },
    {
      "role": "model",
      "parts": [
        "You're likely referring to \"Tandoori Chicken\", which is a popular Indian dish. \n\nHere's the information based on \"Tandoori Chicken\":\n\n```json\n{\n\"Dish Name\": \"Tandoori Chicken\",\n\"Dish Description\": \"Chicken marinated in yogurt and spices, then cooked in a tandoor oven for a smoky, flavorful experience.\",\n\"Cuisine Category\": \"Asia - Indian - North Indian\",\n\"Timing Category\": [\n\"Dinner\",\n\"Lunch\"\n],\n\"Meal Category\": [\n\"Main Course\"\n],\n\"Cooking Method\": [\n\"Grilling\"\n],\n\"Cooking Time\": \"45 minutes\",\n\"Cooking Temperature\": \"Medium Heat (Tandoor Oven)\",\n\"Ingredients\": [\n\"Chicken - 150g\",\n\"Plain Yogurt - 50g\",\n\"Ginger-Garlic Paste - 10g\",\n\"Red Chili Powder - 5g\",\n\"Turmeric Powder - 2g\",\n\"Garam Masala - 3g\",\n\"Cumin Powder - 2g\",\n\"Coriander Powder - 3g\",\n\"Salt - 2g\",\n\"Lemon Juice - 15ml\",\n\"Oil - 5ml\"\n],\n\"Allergic Content\": {\n\"Milk and Dairy Products\": [\n\"Plain Yogurt\"\n]\n},\n\"Serving Size\": \"200g\",\n\"Cooking Steps\": [\n\"1. Marinate chicken pieces with yogurt, ginger-garlic paste, red chili powder, turmeric powder, garam masala, cumin powder, coriander powder, salt, lemon juice, and oil.\",\n\"2. Let the chicken marinate in the refrigerator for at least 2 hours, or preferably overnight.\",\n\"3. Preheat a tandoor oven to medium heat.\",\n\"4. Place the marinated chicken pieces in the tandoor and cook for about 20-25 minutes, turning occasionally, until cooked through and golden brown.\",\n\"5. Serve hot with onion rings, green chutney, and your favorite side dishes.\"\n]\n}\n```\n\nI'm still learning, but I'm getting better at understanding different spellings and variations in dish names. Let me know if you have other dishes you want to explore! \n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "You got it!  I've added the \"Food Category\" to the JSON output, classifying each dish as either \"Vegetarian\" or \"Non-vegetarian\". Here's the updated output for \"Tandoori Chicken\":\n\n```json\n{\n\"Dish Name\": \"Tandoori Chicken\",\n\"Dish Description\": \"Chicken marinated in yogurt and spices, then cooked in a tandoor oven for a smoky, flavorful experience.\",\n\"Cuisine Category\": \"Asia - Indian - North Indian\",\n\"Food Category\": \"Non-vegetarian\",\n\"Timing Category\": [\n\"Dinner\",\n\"Lunch\"\n],\n\"Meal Category\": [\n\"Main Course\"\n],\n\"Cooking Method\": [\n\"Grilling\"\n],\n\"Cooking Time\": \"45 minutes\",\n\"Cooking Temperature\": \"Medium Heat (Tandoor Oven)\",\n\"Ingredients\": [\n\"Chicken - 150g\",\n\"Plain Yogurt - 50g\",\n\"Ginger-Garlic Paste - 10g\",\n\"Red Chili Powder - 5g\",\n\"Turmeric Powder - 2g\",\n\"Garam Masala - 3g\",\n\"Cumin Powder - 2g\",\n\"Coriander Powder - 3g\",\n\"Salt - 2g\",\n\"Lemon Juice - 15ml\",\n\"Oil - 5ml\"\n],\n\"Allergic Content\": {\n\"Milk and Dairy Products\": [\n\"Plain Yogurt\"\n]\n},\n\"Serving Size\": \"200g\",\n\"Cooking Steps\": [\n\"1. Marinate chicken pieces with yogurt, ginger-garlic paste, red chili powder, turmeric powder, garam masala, cumin powder, coriander powder, salt, lemon juice, and oil.\",\n\"2. Let the chicken marinate in the refrigerator for at least 2 hours, or preferably overnight.\",\n\"3. Preheat a tandoor oven to medium heat.\",\n\"4. Place the marinated chicken pieces in the tandoor and cook for about 20-25 minutes, turning occasionally, until cooked through and golden brown.\",\n\"5. Serve hot with onion rings, green chutney, and your favorite side dishes.\"\n]\n}\n```\n\nI'll make sure to include \"Food Category\" in every dish description from now on. Thanks for the suggestion - it's making the output more informative! \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "some in ingredient data generation its give Egg - 1 (50g) or Egg - 1 so please make sure write Egg - weights in g or ml in data not write its 1 or 2 quantity but write in ml or g quanity",
      ],
    },
    {
      "role": "model",
      "parts": [
        "You are absolutely right! Using quantities like \"1 egg\" or \"2 eggs\" isn't very helpful for calculating the total weight of a dish.  \n\nI'll make sure to always provide the weight of ingredients in grams or milliliters, even for things like eggs. For example, I'll use \"Egg - 50g\" instead of \"Egg - 1.\"  \n\nThanks for catching that!  I'll keep this in mind for all future outputs. \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Medu Vada",
      ],
    },
    {
      "role": "model",
      "parts": [
        "```json\n{\n\"Dish Name\": \"Medu Vada\",\n\"Dish Description\": \"Crispy lentil donuts, a popular South Indian breakfast.\",\n\"Cuisine Category\": \"Asia - Indian - South Indian\",\n\"Food Category\": \"Vegetarian\",\n\"Timing Category\": [\n\"Breakfast\",\n\"Lunch\",\n\"Snack\"\n],\n\"Meal Category\": [\n\"Appetizer\",\n\"Snack\",\n\"Finger Food\"\n],\n\"Cooking Method\": [\n\"Frying\"\n],\n\"Cooking Time\": \"20 minutes\",\n\"Cooking Temperature\": \"180°C\",\n\"Ingredients\": [\n\"Urad Dal - 100g\",\n\"Rice - 20g\",\n\"Green Chilies - 5g\",\n\"Ginger - 5g\",\n\"Curry Leaves - 2g\",\n\"Salt - 2g\",\n\"Oil - 50ml\"\n],\n\"Allergic Content\": {\n\"Tree Nuts\": [\n\"Cashews (Optional for Garnish)\"\n]\n},\n\"Serving Size\": \"200g\",\n\"Cooking Steps\": [\n\"1. Soak urad dal and rice in water for at least 4 hours.\",\n\"2. Drain and grind the soaked dal and rice into a smooth paste.\",\n\"3. Add chopped green chilies, ginger, curry leaves, and salt to the batter. Mix well.\",\n\"4. Heat oil in a pan over medium heat.\",\n\"5. Using a spoon, drop spoonfuls of batter into the hot oil. Flatten slightly with a spatula.\",\n\"6. Fry until golden brown and crispy on both sides.\",\n\"7. Remove from oil and drain on paper towels.\",\n\"8. Serve hot with sambar, chutney, or your favorite dipping sauce.\"\n]\n}\n``` \n",
      ],
    }
  ]
)

    
    # Send the dish name to the model and receive the response
    response = chat_session.send_message(dish_name)  

    try:
        dish_info = json.loads(response.text)
        dish_name = dish_info['Dish Name']

        # Check if the dish is already stored in the MongoDB collection
        existing_dish = dish_collection.find_one({"Dish Name": dish_name})

        if existing_dish:
            st.warning(f"The dish '{dish_name}' is already stored in the database.")
            st.subheader("Generated Dish Information:")
            #st.json(existing_dish)  # Display existing dish info
            print(existing_dish)
            dish = existing_dish
        else:
            # Store the new dish information in MongoDB
            dish_collection.insert_one(dish_info)
            st.success(f"The dish '{dish_name}' has been stored in the database.")
            st.subheader("Generated Dish Information:")
            #st.json(dish_info)  # Display the response in JSON format
            print(dish_info)
            dish = dish_info

        # Process each ingredient and convert where necessary
        output_ingredients = []
        total_nutrients = {}

        ingredients = dish["Ingredients"]

        for item in ingredients:
             ingredient_name, quantity_str = item.split(" - ")

             if "ml" in quantity_str:
                 quantity_ml = float(quantity_str.rstrip('ml'))
                 quantity_g = convert_ml_to_g(ingredient_name, quantity_ml, unit_retriever)
                 if quantity_g:
                     output_ingredients.append(f"{ingredient_name} - {quantity_g}g")
                 else:
                     output_ingredients.append(f"{ingredient_name} - {quantity_ml}g")
             else:
                 output_ingredients.append(item)

        for item in output_ingredients:
             ingredient_name, quantity_str = item.split(" - ")
             quantity = float(quantity_str.rstrip('g'))
             ingredient_nutrients = calculate_nutrients(ingredient_name, quantity, ingredient_retriever)
             if ingredient_nutrients:
                 for nutrient, value in ingredient_nutrients.items():
                     if nutrient in total_nutrients:
                         total_nutrients[nutrient] = round(total_nutrients[nutrient] + value, 2)
                     else:
                         total_nutrients[nutrient] = value

        for method in dish["Cooking Method"]:
             total_nut = apply_cooking_retention(total_nutrients, method)

        # Replace the original ingredients with the converted ones
        dish["Ingredients"] = output_ingredients

        # Add the total nutrient content to the dish details
        dish["Total Nutrients"] = total_nut

        # Remove the _id field if it exists
        if "_id" in dish:
             del dish["_id"]

        output_json = json.dumps(dish, indent=4)
    
         # Print the final output
        print(output_json)

        st.json(output_json)


    except json.JSONDecodeError:
        st.error("Failed to decode the response as JSON.")

    
