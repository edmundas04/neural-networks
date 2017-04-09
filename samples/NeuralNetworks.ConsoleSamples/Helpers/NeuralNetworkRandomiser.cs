using System;

namespace NeuralNetworks.ConsoleSamples.Helpers
{
    public static class NeuralNetworkRandomiser
    {
        private static readonly Random _random = new Random();

        public static void Randomise(NeuralNetwork neuralNetwork, double range)
        {
            foreach (var synapseLayer in neuralNetwork.SynapseLayers)
            {
                foreach (var synapse in synapseLayer)
                {
                    synapse.Weight = GetRandomNumber(range);
                    synapse.Target.Bias = GetRandomNumber(range);
                }
            }
        }

        private static double GetRandomNumber(double range)
        {
            var sign = _random.NextDouble() < 0.5D ? -1 : 1;
            return _random.NextDouble() * range * sign;
        }
    }
}
