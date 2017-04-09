using System.Collections.Generic;

namespace NeuralNetworks.Training
{
    public class TrainingElement
    {
        public List<double> Inputs { get; set; }

        public List<double> ExpectedOutputs { get; set; }
    }
}
