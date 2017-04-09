using System;
using System.Collections.Generic;

namespace NeuralNetworks.Extensions
{
    internal static class ListExtensions
    {
        public static void Shuffle<T>(this IList<T> list)
        {
            Random random = new Random(12);
            var count = list.Count;
            while (count > 1)
            {
                --count;
                var  index = random.Next(count + 1);
                var obj = list[index];
                list[index] = list[count];
                list[count] = obj;
            }
        }
    }
}
