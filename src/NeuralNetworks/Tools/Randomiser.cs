using System;

namespace NeuralNetworks.Tools
{
    public static class Randomiser
    {
        public static void Randomise(NeuralNetworkDto dto, double range)
        {
            Randomise(dto, range, new Random());
        }

        public static void Randomise(NeuralNetworkDto dto, double range, Random random)
        {
            foreach (var synapseLayer in dto.SynapsesLayers)
            {
                foreach (var synapse in synapseLayer)
                {
                    synapse.Weight = GetRandomNumber(range, random);
                }
            }

            foreach (var neuronsLayer in dto.NeuronsLayers)
            {
                foreach (var neuron in neuronsLayer)
                {
                    neuron.Bias = GetRandomNumber(range, random);
                }
            }
        }

        private static double GetRandomNumber(double range, Random random)
        {
            var sign = random.NextDouble() < 0.5D ? -1 : 1;
            return random.NextDouble() * range * sign;
        }
    }
}
