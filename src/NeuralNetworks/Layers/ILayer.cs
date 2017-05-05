namespace NeuralNetworks.Layers
{
    public interface ILayer
    {
        void UpdateSynapsesWeights(double[] newSynapsesWeighs);
        void UpdateNeuronsBiases(double[] newNeuronsBiases);
        double[] ProduceActivation(double[] input);
    }
}
