using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

// Image resizing (Windows-friendly)
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

class Program
{
    private const int MAX_TURN_ANGLE = 180;
    private const int MIN_TURN_ANGLE = -180;

    // Resize / compress settings
    private const int MAX_DIMENSION = 640;   // max width/height after resize
    private const long JPEG_QUALITY = 80;    // 1..100 (lower = smaller)

    static async Task Main(string[] args)
    {
        // Usage:
        //   dotnet run -- car-on-road.png
        // Ollama must be running:
        //   ollama serve
        //   ollama pull llava

        string model = "llava";

        string imagePath = args.Length > 0 ? args[0] : "car-on-road.png";
        if (!File.Exists(imagePath))
        {
            Console.WriteLine($"ERROR: Image file not found: {imagePath}");
            return;
        }

        // Prompt: numeric-only angle output (easy to validate)
        string prompt =
            "Based on the image, return ONLY one numeric steering turn angle in degrees.\n" +
            "The value must be between -180 and 180.\n" +
            "Do not include words, units, punctuation, or explanation.\n" +
            "Examples of valid output:\n" +
            "-15\n" +
            "0\n" +
            "32.5\n";

        // Convert to small JPEG bytes (resized + compressed) to reduce payload
        byte[] jpegBytes;
        try
        {
            jpegBytes = LoadResizeAndCompressToJpeg(imagePath, MAX_DIMENSION, JPEG_QUALITY);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR: Failed to resize/compress image: {ex.Message}");
            return;
        }

        string base64Image = Convert.ToBase64String(jpegBytes);

        var requestBody = new OllamaRequest(
            model: model,
            prompt: prompt,
            images: new[] { base64Image },
            stream: false
        );

        using HttpClient client = new();
        client.Timeout = TimeSpan.FromMinutes(2);

        string json = JsonSerializer.Serialize(requestBody, OllamaContext.Default.OllamaRequest);
        Console.WriteLine($"Sending request to Ollama... (JSON size: {json.Length} bytes)");
        Console.WriteLine($"Image payload: {jpegBytes.Length} bytes (JPEG), base64 chars: {base64Image.Length}");

        var content = new StringContent(json, Encoding.UTF8, "application/json");

        try
        {
            HttpResponseMessage response = await client.PostAsync("http://localhost:11434/api/generate", content);

            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine($"ERROR: Ollama returned status code {response.StatusCode}");
                try
                {
                    string errBody = await response.Content.ReadAsStringAsync();
                    if (!string.IsNullOrWhiteSpace(errBody))
                        Console.WriteLine($"Server Message: {errBody}");
                }
                catch { }
                return;
            }

            string responseJson = await response.Content.ReadAsStringAsync();
            using JsonDocument doc = JsonDocument.Parse(responseJson);

            string llmResponse = doc.RootElement.GetProperty("response").GetString() ?? "";
            Console.WriteLine("\nRaw LLM response:");
            Console.WriteLine(llmResponse);

            // Validate numeric output
            if (!TryParseAngle(llmResponse, out double angle))
            {
                Console.WriteLine("\nFAIL: Output was not a numeric value only.");
                return;
            }

            if (angle > MAX_TURN_ANGLE || angle < MIN_TURN_ANGLE)
            {
                Console.WriteLine($"\nFAIL: Angle out of range ({MIN_TURN_ANGLE}..{MAX_TURN_ANGLE}). Value: {angle}");
                return;
            }

            Console.WriteLine($"\nPASS: Steering angle = {angle}");
        }
        catch (HttpRequestException ex)
        {
            Console.WriteLine("\nCRITICAL ERROR: Could not connect to Ollama or the connection was reset.");
            Console.WriteLine("Possible causes:");
            Console.WriteLine("1) Ollama is not running (run 'ollama serve').");
            Console.WriteLine("2) Ollama crashed while processing the image (check Ollama logs).");
            Console.WriteLine("3) The 'llava' model is corrupted (try 'ollama rm llava' then 'ollama pull llava').");
            Console.WriteLine($"\nTechnical Details: {ex.Message}");
        }
    }

    private static bool TryParseAngle(string s, out double angle)
    {
        angle = 0;

        // Must be *only* a number (allow whitespace around it)
        string trimmed = (s ?? "").Trim();

        // Reject if it contains spaces/newlines inside (common when it adds words)
        // But allow leading/trailing whitespace.
        if (trimmed.Contains(' ') || trimmed.Contains('\n') || trimmed.Contains('\r') || trimmed.Contains('\t'))
            return false;

        return double.TryParse(trimmed, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out angle);
    }

    /// <summary>
    /// Loads an image from disk, scales it down so max(width,height)=maxDim (if needed),
    /// then encodes as JPEG with given quality.
    /// </summary>
    private static byte[] LoadResizeAndCompressToJpeg(string path, int maxDim, long jpegQuality)
    {
        using var original = Image.FromFile(path);

        int ow = original.Width;
        int oh = original.Height;

        // No resize needed?
        if (ow <= maxDim && oh <= maxDim)
        {
            // Still recompress to JPEG to keep payload small even if PNG
            using var msNoResize = new MemoryStream();
            SaveJpeg(original, msNoResize, jpegQuality);
            return msNoResize.ToArray();
        }

        // Compute scale
        double scale = ow > oh ? (double)maxDim / ow : (double)maxDim / oh;
        int nw = Math.Max(1, (int)Math.Round(ow * scale));
        int nh = Math.Max(1, (int)Math.Round(oh * scale));

        using var resized = new Bitmap(nw, nh);
        resized.SetResolution(original.HorizontalResolution, original.VerticalResolution);

        using (var g = Graphics.FromImage(resized))
        {
            g.CompositingQuality = CompositingQuality.HighQuality;
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.SmoothingMode = SmoothingMode.HighQuality;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;

            g.DrawImage(original, 0, 0, nw, nh);
        }

        using var ms = new MemoryStream();
        SaveJpeg(resized, ms, jpegQuality);
        return ms.ToArray();
    }

    private static void SaveJpeg(Image image, Stream output, long quality)
    {
        quality = Math.Clamp(quality, 1, 100);

        ImageCodecInfo? jpegCodec = null;
        foreach (var c in ImageCodecInfo.GetImageEncoders())
        {
            if (c.MimeType.Equals("image/jpeg", StringComparison.OrdinalIgnoreCase))
            {
                jpegCodec = c;
                break;
            }
        }

        if (jpegCodec == null)
            throw new InvalidOperationException("JPEG encoder not found on this system.");

        using var encParams = new EncoderParameters(1);
        encParams.Param[0] = new EncoderParameter(Encoder.Quality, quality);

        image.Save(output, jpegCodec, encParams);
    }
}

public record OllamaRequest(string model, string prompt, string[] images, bool stream);

[JsonSerializable(typeof(OllamaRequest))]
public partial class OllamaContext : JsonSerializerContext { }
