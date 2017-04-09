namespace NeuralNetworks
{
    public class Synapse
    {
        public Neuron Primary { get; set; }
        public Neuron Target { get; set; }
        public double Weight { get; set; }

        public Synapse(Neuron primary, Neuron target, double weight)
        {
            Primary = primary;
            Target = target;
            Weight = weight;
        }
    }
}
