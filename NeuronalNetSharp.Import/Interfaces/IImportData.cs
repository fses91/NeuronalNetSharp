namespace NeuronalNetSharp.Import.Interfaces
{
    internal interface IImportData<out T>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="dataFile">The data file.</param>
        /// <param name="labelFile">The label file</param>
        /// <param name="amountData">The amount of data you want to import.</param>
        /// <returns>The imported data.</returns>
        T ImportData(string dataFile, string labelFile, int amountData);
    }
}
