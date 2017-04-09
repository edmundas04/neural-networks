using System.Collections.Generic;

namespace NeuralNetworks.Training
{
    public interface ITrainer
    {
        void Train(NeuralNetwork neuralNetwork, List<TrainingElement> trainingData);
    }
}
