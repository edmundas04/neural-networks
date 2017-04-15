namespace NeuralNetworks.ConsoleSamples.Examples
{
    public interface IExample
    {
        NeuralNetworkDto CreateNeuralNetwork();
        void Train(NeuralNetworkDto neuralNetwork);
        void DisplayEvaluation(NeuralNetworkDto neuralNetwork);
    }
}
