using System.Collections.Generic;

namespace NeuronalNetSharp.Import.Interfaces
{
    internal interface IImportData
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="dataFile">The data file.</param>
        /// <param name="labelFile">The label file</param>
        /// <returns>The imported data.</returns>
        ICollection<IDataset> ImportData(string dataFile, string labelFile);
    }
}
