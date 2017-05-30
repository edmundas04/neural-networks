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
        int PrimaryNeuronsCount { get; }

        void Produce(double[] input);
    }
}
