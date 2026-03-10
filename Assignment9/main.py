# Imports
import requests

# -------------------- Configuration --------------------
# Run 'ollama list' in your terminal to see installed model names.
# Common names: deepseek-r1, deepseek-r1:7b, deepseek-r1:1.5b
MODEL = "deepseek-r1:8b"

# -------------------- User Input Handling --------------------
def get_user_input():
	print("=" * 70)
	print("PERSONALIZED DINNER RECIPE GENERATOR (LLM - OLLAMA/DEEPSEEK)")
	print("=" * 70)
	# Improved input handling: require non-empty input, allow 'none'
	while True:
		allergies = input("Enter your allergies (comma-separated, or 'none'): ").strip()
		if allergies:
			break
		print("Allergies cannot be empty. Please enter 'none' if you have no allergies.")
	while True:
		preferences = input("Enter your dietary preferences (e.g., vegetarian, gluten-free, keto, or 'none'): ").strip()
		if preferences:
			break
		print("Preferences cannot be empty. Please enter 'none' if you have no preferences.")
	# Normalize input
	allergies = allergies.lower()
	preferences = preferences.lower()
	return allergies, preferences

# -------------------- Prompt Construction --------------------
def build_prompt(allergies, preferences):
	"""
	Construct a prompt for the LLM with instruction, example, and CoT components.
	"""
	prompt = f"""
You are a helpful AI chef. Your task is to create a personalized dinner recipe for a user.

Instructions:
- The recipe must strictly avoid these allergies: {allergies}
- The recipe must follow these dietary preferences: {preferences}
- Output a recipe name, a filtered ingredients list, and step-by-step instructions.

Example:
User Allergies: peanuts, shellfish
User Preferences: vegetarian

Recipe Name: Creamy Mushroom Risotto
Ingredients:
- Arborio rice
- Mushrooms
- Vegetable broth
- Parmesan cheese
- Onion
- Garlic
- Olive oil
- Salt
- Pepper

Instructions:
1. Sauté onions and garlic in olive oil.
2. Add mushrooms and cook until soft.
3. Stir in rice, then gradually add broth, stirring until creamy.
4. Mix in parmesan, salt, and pepper to taste.

Chain-of-Thought:
First, think about what ingredients are safe and fit the preferences. Then, plan a recipe that uses only those ingredients. Finally, write clear, step-by-step instructions.

Now, generate a new recipe for:
Allergies: {allergies}
Preferences: {preferences}
"""
	return prompt

# -------------------- LLM Query Function --------------------
def query_llm(prompt):
	"""
	Query the local Ollama server running the DeepSeek model.
	"""
	url = "http://localhost:11434/api/generate"
	payload = {
		"model": MODEL,
		"prompt": prompt,
		"stream": False
	}
	print("\nQuerying Ollama/DeepSeek LLM... (this may take a few seconds)")
	try:
		response = requests.post(url, json=payload, timeout=120)
		response.raise_for_status()
		result = response.json()
		return result.get("response", "").strip()
	except Exception as e:
		print(f"Error querying Ollama: {e}")
		return "LLM query failed."

# -------------------- Save Output --------------------
def save_output(prompt, response, filename="recipe_output.txt"):
	"""
	Save the prompt and LLM response to a .txt file.
	"""
	with open(filename, "w", encoding="utf-8") as f:
		f.write("Prompt:\n")
		f.write(prompt)
		f.write("\n\nLLM Response:\n")
		f.write(response)
	print(f"\nPrompt and response saved to {filename}")

# -------------------- Main Workflow --------------------
def main():
	allergies, preferences = get_user_input()
	prompt = build_prompt(allergies, preferences)
	response = query_llm(prompt)
	print("\n--- LLM Recipe Response ---\n")
	print(response)
	save_output(prompt, response)

if __name__ == "__main__":
	main()
