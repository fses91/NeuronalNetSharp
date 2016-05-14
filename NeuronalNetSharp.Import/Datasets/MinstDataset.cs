using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuronalNetSharp.Import.Interfaces;

namespace NeuronalNetSharp.Import.Datasets
{
    public class MinstDataset : IDataset<int[][]>
    {
        public string Label { get; set; }

        public int[][] Data { get; set; }
    }
}
