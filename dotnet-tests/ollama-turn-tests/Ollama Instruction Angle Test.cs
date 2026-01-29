using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

class Program
{
    private const int MAX_TURN_ANGLE = 180;
    private const int MIN_TURN_ANGLE = -180;

    static async Task Main()
    {
        // you have to have an Ollama server running with the llava model for this test to work
        // ollama serve
        // ollama pull llava

        string model = "llava";
        string prompt =
            "Based on the image, return ONLY one numeric steering turn angle in degrees.\n" +
            "Do not include words, units, or explanation.";

        string imagePath = "car_on_road.jpg"; // placeholder name

        if (!File.Exists(imagePath))
        {
            Console.WriteLine("ERROR: Image file not found.");
            return;
        }

        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string base64Image = Convert.ToBase64String(imageBytes);

        var requestBody = new
        {
            model = model,
            prompt = prompt,
            images = new[] { base64Image },
            stream = false
        };

        using HttpClient client = new();
        string json = JsonSerializer.Serialize(requestBody);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        HttpResponseMessage response = await client.PostAsync(
            "http://localhost:11434/api/generate",
            content
        );

        if (!response.IsSuccessStatusCode)
        {
            Console.WriteLine("ERROR: Failed to contact Ollama.");
            return;
        }

        string responseJson = await response.Content.ReadAsStringAsync();
        using JsonDocument doc = JsonDocument.Parse(responseJson);

        string llmResponse = doc.RootElement.GetProperty("response").GetString() ?? "";

        Console.WriteLine("Response from Ollama LLM:");
        Console.WriteLine(llmResponse);

        // Attempt to parse numeric angle
        if (!double.TryParse(llmResponse.Trim(), out double angle))
        {
            Console.WriteLine("FAIL: Output was not a numeric value.");
            Environment.Exit(1);
        }

        if (angle > MAX_TURN_ANGLE || angle < MIN_TURN_ANGLE)
        {
            Console.WriteLine("FAIL: The turn angle is out of valid turn range.");
        }
        else
        {
            Console.WriteLine("PASS: The turn angle is within the valid turn range.");
        }
    }
}
