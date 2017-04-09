namespace NeuralNetworks
{
    public class Synapse
    {
        public int Id { get; }

        public double Weight { get; set; }

        public Neuron Primary { get; set; }

        public Neuron Target { get; set; }

        public Synapse(int id)
        {
            Id = id;
        }
    }
}
