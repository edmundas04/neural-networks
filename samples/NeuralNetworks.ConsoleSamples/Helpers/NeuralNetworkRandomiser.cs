using System;

namespace NeuralNetworks.ConsoleSamples.Helpers
{
    public static class NeuralNetworkRandomiser
    {
        private static readonly Random _random = new Random(5);

        public static void Randomise(NeuralNetworkDto dto, double range)
        {
            foreach (var synapseLayer in dto.SynapsesLayers)
            {
                foreach (var synapse in synapseLayer)
                {
                    synapse.Weight = GetRandomNumber(range);
                }
            }

            foreach (var neuronsLayer in dto.NeuronsLayers)
            {
                foreach (var neuron in neuronsLayer)
                {
                    neuron.Bias = GetRandomNumber(range);
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
