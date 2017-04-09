using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.ConsoleSamples.Examples
{
    public interface IExample
    {
        NeuralNetwork CreateNeuralNetwork();
        void Train(NeuralNetwork neuralNetwork);
        void DisplayEvaluation(NeuralNetwork neuralNetwork);
    }
}
