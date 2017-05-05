using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.Exceptions
{
    public class NeuralNetworksException : Exception
    {
        public NeuralNetworksException() : base()
        {

        }

        public NeuralNetworksException(string message) : base(message)
        {

        }
    }
}
