using System.Collections.Generic;

namespace NeuralNetworks
{
    public class NeuralNetworkDto
    {
        public int InputNeuronsCount { get; set; }
        public List<List<NeuronDto>> NeuronsLayers { get; set; }
        public List<List<SynapseDto>> SynapsesLayers { get; set; }
        public ActivationFunctionType ActivationFunctionType { get; set; }
    }

    public class NeuronDto
    {
        public double Bias { get; set; }
        public int Position { get; set; }
    }
    
    public class SynapseDto
    {
        public int PrimaryNeuronPosition { get; set; }
        public int TargetNeuronPosition { get; set; }
        public double Weight { get; set; }
    }

    public enum ActivationFunctionType
    {
        Sigmoid
    }
}
