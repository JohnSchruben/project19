import ollama

client = ollama.Client()

model = "llava"  # May change the model
prompt = (
    "Based on the image, return ONLY one numeric steering turn angle in degrees.\n"
    "Do not include words, units, or explanation."
)


response = client.generate(
    model=model,
    prompt=prompt,
    images=["car_on_road.jpg"] #car_on_road.jpg is a placeholder name
)

print("Response from Ollama LLM:")
print(response["response"])

try:
    angle = float(response["response"].strip()) # Convert to float

# Handle non-numeric output
# Ex: "Turn angle is 45 degrees", "forty five", and "45 degrees" are all invalid responses
except ValueError:
    print("FAIL: Output was not a numeric value.")
    exit(1)

if angle > 180 or angle < -180:
    print("FAIL: The turn angle is out of valid turn range.")
else:
    print("PASS: The turn angle is within the valid turnrange.")
