using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuronalNetSharp.Import.Interfaces
{
    /// <summary>
    /// An interface for the dataset.
    /// </summary>
    /// <typeparam name="T">The type of the data.</typeparam>
    public interface IDataset
    {
        /// <summary>
        /// The label for the data.
        /// </summary>
        string Label { get; }

        /// <summary>
        /// The amount of rows of the input data. 
        /// </summary>
        int Rows { get; }

        /// <summary>
        /// The amount of columns of the input data.
        /// </summary>
        int Columns { get; }

        /// <summary>
        /// The data.
        /// </summary>
        Matrix Data { get; }
    }
}
