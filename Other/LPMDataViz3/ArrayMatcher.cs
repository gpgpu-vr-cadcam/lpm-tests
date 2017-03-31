using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace LPMDataViz
{
    class ArrayMatcher
    {
        public int[] Lenghts;
        public int[] MaxIP;
        public int[] MinIP;

        public int[] Array;

        public int Count;
        public int MaxLenght;


        public void LoadMasksFromFile(int count, string path)
        {
            Count = count;
            Lenghts = new int[Count];
            MaxIP = new int[Count];
            MinIP = new int[Count];

            using (StreamReader file = new StreamReader(path))
            {
                string line;
                for (int mask = 0; mask < Count; ++mask)
                {
                    line = file.ReadLine();

                    int address = 0;
                    var parts = line.Split(new char[] { ';', '.' });
                    
                    for (int i = 0; i < 4; ++i)
                        address |= int.Parse(parts[2 + i]) << (8 * (3 - i));

                    Lenghts[mask] = int.Parse(parts[6]);

                    MinIP[mask] = address;
                    MaxIP[mask] = ((1 << (32 - Lenghts[mask])) - 1) | address;
                }
            }

        }

        public void PrintMasks()
        {
            StreamWriter outFile = new StreamWriter("outFile.txt");
            for (int mask = 0; mask < Count; ++mask)
            {
                outFile.Write("{0}\n{1} . ", Convert.ToString(MinIP[mask], 2).PadLeft(32, '0'), Convert.ToString(MaxIP[mask], 2).PadLeft(32, '0'));
                outFile.WriteLine(Lenghts[mask]);
            }

            outFile.Close();
        }

        public void Build()
        {
            MaxLenght = Lenghts.Max();

            //Array = new int[1 << MaxLenght];
            //for (int i = 0; i < Array.Length; ++i)
            //            Array[i] = -1;

            List<Entry> entries = new List<Entry>();
            List<Entry> longMasks = new List<Entry>();

            for(int i = 0; i < Count; ++i)
                if(Lenghts[i] <= 24)
                    entries.Add(new Entry(MaxIP[i], MinIP[i], Lenghts[i], i));
                else
                    longMasks.Add(new Entry(MaxIP[i], MinIP[i], Lenghts[i], i));

            entries.Sort();

            MaxLenght = Math.Min(Lenghts.Max(), 24);

            Array = new int[1 << MaxLenght];
            for (int i = 0; i < Array.Length; ++i)
                Array[i] = -1;

            //foreach (var entry in entries)
            //{
            //    for (int i = entry.MinIP; i < entry.MaxIP; ++i)
            //        Array[(i >> (32 - MaxLenght)) & ((1 << 24)-1)] = entry.Index;
            //}
        }
    }

    class Entry : IComparable<Entry>
    {
        public int MaxIP;
        public int MinIP;
        public int Lenght;
        public int Index;

        public Entry(int maxIp, int minIp, int lenght, int index)
        {
            MaxIP = maxIp;
            MinIP = minIp;
            Lenght = lenght;
            Index = index;
        }

        public int CompareTo(Entry other)
        {
            int r = -Lenght.CompareTo(other.Lenght);

            if (r == 0)
                r = MinIP.CompareTo(other.MinIP);

            if (r == 0)
                r = MaxIP.CompareTo(other.MaxIP);

            return r;
        }

        public override string ToString()
        {
            return Lenght.ToString() + " " + MinIP.ToString() + " " + MaxIP.ToString() + " " + Lenght.ToString();
        }
    }
}
