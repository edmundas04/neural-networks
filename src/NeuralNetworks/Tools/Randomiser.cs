using System;
using System.Linq;

namespace NeuralNetworks.Tools
{
    public static class Randomiser
    {
        public static void Randomise(NeuralNetworkDto dto)
        {
            Randomise(dto, new Random());
        }

        public static void Randomise(NeuralNetworkDto dto, Random random)
        {
            foreach (var synapseLayer in dto.SynapsesLayers)
            {
                var primaryNeuronsCount = synapseLayer.Max(x => x.PrimaryNeuronPosition) + 1;
                var sqrt = Math.Sqrt(primaryNeuronsCount);

                foreach (var synapse in synapseLayer)
                {
                    synapse.Weight = GetRandomNumber(random) / sqrt;
                }
            }

            foreach (var neuronsLayer in dto.NeuronsLayers)
            {
                foreach (var neuron in neuronsLayer)
                {
                    neuron.Bias = GetRandomNumber(random);
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
