using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.Layers;
using System.Linq;

namespace NeuralNetworks.ConsoleSamples.Extensions
{
    public static class NeuralNetworkDtoExtensions
    {
        public static NeuralNetworkDto Copy(this NeuralNetworkDto neuralNetworkDto)
        {
            return new NeuralNetworkDto
            {
                ActivationFunctionType = neuralNetworkDto.ActivationFunctionType,
                InputNeuronsCount = neuralNetworkDto.InputNeuronsCount,
                NeuronsLayers = neuralNetworkDto.NeuronsLayers.Select(s => s.Select(x => x.Copy()).ToList()).ToList(),
                SynapsesLayers = neuralNetworkDto.SynapsesLayers.Select(s => s.Select(x => x.Copy()).ToList()).ToList()
            };
        }

        private static SynapseDto Copy(this SynapseDto synapseDto)
        {
            return new SynapseDto
            {
                PrimaryNeuronPosition = synapseDto.PrimaryNeuronPosition,
                TargetNeuronPosition = synapseDto.TargetNeuronPosition,
                Weight = synapseDto.Weight
            };
        }

        private static NeuronDto Copy(this NeuronDto neuronDto)
        {
            return new NeuronDto
            {
                Bias = neuronDto.Bias,
                Position = neuronDto.Position
            };
        }

        //TODO: Remove this code after refactoring
        public static ILayer[] ToLayers(this NeuralNetworkDto neuralNetworkDto)
        {
            var result = new ILayer[neuralNetworkDto.NeuronsLayers.Count];
            var primaryNeuronsCount = neuralNetworkDto.InputNeuronsCount;

            for (int i = 0; i < neuralNetworkDto.NeuronsLayers.Count; i++)
            {
                var neurons = neuralNetworkDto.NeuronsLayers[i];
                var synapses = neuralNetworkDto.SynapsesLayers[i];

                var layer = new FullyConnectedLayer(new Sigmoid(), synapses.Select(s => s.Weight).ToArray(), neurons.Select(s => s.Bias).ToArray());
                result[i] = layer;
                primaryNeuronsCount = neurons.Count;
            }

            return result;
        }
    }
}
