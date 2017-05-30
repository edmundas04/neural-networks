using NeuralNetworks.Layers;
using System;
using System.Linq;

namespace NeuralNetworks.Tools
{
    public static class Randomiser
    {
        public static void Randomise(ILayer[] layers)
        {
            Randomise(layers, new Random());
        }

        public static void Randomise(ILayer[] layers, Random random)
        {
            foreach (var layer in layers)
            {
                var sqrt = Math.Sqrt(layer.PrimaryNeuronsCount);

                for (int i = 0; i < layer.SynapsesWeights.Length; i++)
                {
                    layer.SynapsesWeights[i] = GetRandomNumber(random) / sqrt;
                }
            }

            foreach (var layer in layers)
            {
                for (int i = 0; i < layer.NeuronsBiases.Length; i++)
                {
                    layer.NeuronsBiases[i] = GetRandomNumber(random);
                }
            }
        }

        private static double GetRandomNumber(Random random)
        {
            var sign = random.NextDouble() < 0.5D ? -1 : 1;
            return random.NextDouble() * sign;
        }
    }
}
