//Important! Install 'Azure.AI.Vision.ImageAnalysis' NuGet package before run this file
using Azure;
using Azure.AI.Vision.ImageAnalysis;

//No es buena práctica poner en el código estos datos, sino mediante variables de entorno
//pero lo hacemos así a modo de prueba
string endpoint = "<your endpoint>";
string key = "<your key>";

// Crear un cliente de Análisis de Imagen.
ImageAnalysisClient client = new ImageAnalysisClient(
    new Uri(endpoint),
    new AzureKeyCredential(key));

// Usar la URL de la imagen.
Uri imageURL = new Uri("https://historia.nationalgeographic.com.es/medio/2020/01/22/churchill-con-su-famoso-sombrero-de-copa-en-una-foto-tomada-en-1945-cuando-la-segunda-guerra-mundial-llegaba-a-su-fin_73fb69b8_800x699.jpg");

//Todas las herramientas que usaremos
VisualFeatures visualFeatures =
    VisualFeatures.Caption |
    VisualFeatures.DenseCaptions |
    VisualFeatures.Objects |
    VisualFeatures.Read |
    VisualFeatures.Tags |
    VisualFeatures.People |
    VisualFeatures.SmartCrops;

ImageAnalysisOptions options = new ImageAnalysisOptions
{
    GenderNeutralCaption = true, //No asume el género de las personas
    Language = "en",
    SmartCropsAspectRatios = new float[] { 0.9F, 1.33F }
};


ImageAnalysisResult result = client.Analyze(
    imageURL,
    visualFeatures,
    options);

Console.WriteLine("Resultados del analisis de imagen");

// Mostrar la descripción en la consola
Console.WriteLine("Descripción:");
Console.WriteLine($"   '{result.Caption.Text}', Confidence {result.Caption.Confidence:F4}");

// Mostrar en la consola los resultados de las descripciones detalladas
Console.WriteLine(" Dense Captions:");
foreach (DenseCaption denseCaption in result.DenseCaptions.Values)
{
    Console.WriteLine($"   '{denseCaption.Text}', Confidence {denseCaption.Confidence:F4}, Bounding box {denseCaption.BoundingBox}");
}

// Mostrar los resultados de la detección de objetos en la consola
Console.WriteLine(" Objects:");
foreach (DetectedObject detectedObject in result.Objects.Values)
{
    Console.WriteLine($"   '{detectedObject.Tags.First().Name}', Bounding box {detectedObject.BoundingBox.ToString()}");
}

// Mostrar los resultados del análisis de texto (OCR) en la consola
Console.WriteLine(" Read:");
foreach (DetectedTextBlock block in result.Read.Blocks)
    foreach (DetectedTextLine line in block.Lines)
    {
        Console.WriteLine($"   Line: '{line.Text}', Bounding Polygon: [{string.Join(" ", line.BoundingPolygon)}]");
        foreach (DetectedTextWord word in line.Words)
        {
            Console.WriteLine($"     Word: '{word.Text}', Confidence {word.Confidence.ToString("#.####")}, Bounding Polygon: [{string.Join(" ", word.BoundingPolygon)}]");
        }
    }

// Mostrar los resultados de las etiquetas en la consola
Console.WriteLine(" Tags:");
foreach (DetectedTag tag in result.Tags.Values)
{
    Console.WriteLine($"   '{tag.Name}', Confidence {tag.Confidence:F4}");
}

// Mostrar los resultados de las personas en la consola
Console.WriteLine(" People:");
foreach (DetectedPerson person in result.People.Values)
{
    Console.WriteLine($"   Person: Bounding box {person.BoundingBox.ToString()}, Confidence {person.Confidence:F4}");
}

// Imprimir las sugerencias de los recortes inteligentes
Console.WriteLine(" SmartCrops:");
foreach (CropRegion cropRegion in result.SmartCrops.Values)
{
    Console.WriteLine($"   Aspect ratio: {cropRegion.AspectRatio}, Bounding box: {cropRegion.BoundingBox}");
}

// Imprimir los metadatos
Console.WriteLine(" Metadata:");
Console.WriteLine($"   Model: {result.ModelVersion}");
Console.WriteLine($"   Image width: {result.Metadata.Width}");
Console.WriteLine($"   Image hight: {result.Metadata.Height}");