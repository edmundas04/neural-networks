using NeuralNetworks.Training;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;
using System.Reflection;

namespace NeuralNetworks.ConsoleSamples.Helpers
{
    public static class TrainingDataLoader
    {
        public static List<TrainingElement> Load(string resourceName)
        {
            using (Stream manifestResourceStream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName))
            {
                using (StreamReader streamReader = new StreamReader(manifestResourceStream))
                {
                    return JsonConvert.DeserializeObject<List<TrainingElement>>(streamReader.ReadToEnd());
                }
            }
        }
    }
}
