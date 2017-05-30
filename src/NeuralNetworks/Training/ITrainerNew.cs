using System.Collections.Generic;

namespace NeuralNetworks.Training
{
    public interface ITrainerNew
    {
        void Train(List<TrainingElement> trainingData);
    }
}
