using System.Collections.Generic;

namespace NeuralNetworks.Training
{
    public interface ITrainer
    {
        void Train(List<TrainingElement> trainingData);
    }
}
