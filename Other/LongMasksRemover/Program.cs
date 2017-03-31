using System;
using System.IO;
using System.Linq;

namespace LongMasksRemover
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = @"C:\Users\alber\OneDrive\MiNI\Projekty\LPM\TestData\";

            foreach (var file in Directory.GetFiles(path).Select(p => new
                {
                    Path = path,
                    InFile = new StreamReader(p),
                    OutFile = new StreamWriter(Path.Combine(Path.GetDirectoryName(p), Path.GetFileNameWithoutExtension(p) +"_short_masks" + Path.GetExtension(p)))
                })
            )
            {
                string line;
                while ((line = file.InFile.ReadLine()) != null)
                {
                    var parts = line.Split(new char[] { ';', '.' });

                    if (int.Parse(parts[6]) <= 24)
                        file.OutFile.WriteLine(line);
                }

                file.OutFile.Close();
                file.InFile.Close();
            }
        }
    }
}
