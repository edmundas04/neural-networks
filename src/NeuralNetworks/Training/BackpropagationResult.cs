using System.Collections.Generic;

namespace NeuralNetworks.Training
{
    public class BackpropagationResult
    {
        public List<BackpropagationItem> BiasesGradients { get; set; }
        public List<BackpropagationItem> WeightsGradients { get; set; }
    }

    public class BackpropagationItem
    {
        public int Id { get; set; }
        public double Gradient { get; set; }
    }
}
