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
    }
}
