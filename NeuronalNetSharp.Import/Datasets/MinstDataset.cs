using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuronalNetSharp.Import.Interfaces;

namespace NeuronalNetSharp.Import.Datasets
{
    public class MinstDataset : IDataset
    {
        public MinstDataset(Matrix data, string label, int rows, int columns)
        {
            Data = data;
            Label = label;
            NumberOfRows = rows;
            NumberOfColumns = columns;
        }

        public string Label { get; }

        public int NumberOfRows { get; }

        public int NumberOfColumns { get; }

        public Matrix Data { get; set; }
    }
}
