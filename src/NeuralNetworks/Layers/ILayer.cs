using NeuralNetworks.ActivationFunctions;

namespace NeuralNetworks.Layers
{
    public interface ILayer
    {
        IActivationFunction ActivationFunction { get; }
        double[] Outputs { get; }
        double[] Activations { get; }
        double[] SynapsesWeights { get; }
        double[] NeuronsBiases { get; }
        
        void Produce(double[] input);
    }
}
