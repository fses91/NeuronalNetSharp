namespace NeuronalNetSharp.Import.Interfaces
{
    internal interface IImportData<out T>
    {
        T ImportData(string dataFile, string labelFile, int amountData);
    }
}
