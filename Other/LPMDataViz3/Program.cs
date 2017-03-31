using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace LPMDataViz
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = @"C:\Users\alber\OneDrive\MiNI\Projekty\LPM\TestData\data-raw-table_australia_012016.txt";

            ////Building tree
            //Tree t = new Tree();
            //int[] r = new[] {5, 5, 6, 8, 8};

            //t.LoadMasksFromFile(10000, r, path);
            //t.PrintMasks();

            //t.Build();

            ////Matching ip's
            //var s = t.Match(t.Masks, t.Count);
            //Console.WriteLine("{0}",  s.Sum());

            //Building array
            ArrayMatcher a = new ArrayMatcher();

            a.LoadMasksFromFile(500000, path);
            //a.PrintMasks();

            a.Build();

            //Matching ip's
        }
    }
}
