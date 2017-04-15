using System.Collections.Generic;

namespace NeuralNetworks.Training
{
    public interface ITrainer
    {
        void Train(NeuralNetworkDto neuralNetwork, List<TrainingElement> trainingData);
    }
}
