using System.Collections.Generic;

namespace NeuralNetworks.Running
{
    public interface IRunner
    {
        List<double> Run(NeuralNetwork neuralNetwork, List<double> inputValues);
    }
}
