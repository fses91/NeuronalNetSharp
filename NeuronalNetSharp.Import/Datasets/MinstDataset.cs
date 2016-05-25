using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Single;
using NeuronalNetSharp.Import.Interfaces;

namespace NeuronalNetSharp.Import.Datasets
{
    public class MinstDataset : IDataset
    {
        public MinstDataset(Matrix data, string label, int rows, int columns)
        {
            Data = data;
            Label = label;
            Rows = rows;
            Columns = columns;
        }

        public string Label { get; }

        public int Rows { get; }

        public int Columns { get; }

        public Matrix Data { get; }
    }
}
