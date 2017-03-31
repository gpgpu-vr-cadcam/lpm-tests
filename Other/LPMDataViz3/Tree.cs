using System;
using System.IO;
using System.Linq;
using System.Text;
using MoreLinq;

namespace LPMDataViz
{
    public class Tree
    {
        //Number of masks stored in tree structure
        public int Count;

        //Bits per level
        public int[] R;
        public int[] rSums; //inclusive scan
        public int[] rPreSums; //exclusive scan

        //Number of levels
        public int L;

        //Masks stored in tree structure
        public int[][] Masks;
        public int[] Lenghts;

        //Number of nodes on each tree level
        public int[] LevelsSizes;

        //Number of possible children of tree node
        public int[] ChildrenCount;

        //Children arrays for tree nodes
        public int[][] Children;

        //Lists items (indexes to Masks array)
        public int[] ListItems;

        //Start indexes of lists (int ListsItems array) of tree nodes
        public int[][] ListsStarts;

        //Lenghts of lists of tree nodes
        public int[][] ListsLenghts;

        public void LoadMasksFromFile(int count, int[] r, string path)
        {
            R = r;
            rSums = R.Scan((a, b) => a + b).ToArray();
            rPreSums = R.PreScan((a, b) => a + b, 0).ToArray();

            Count = count;
            L = R.Length;

            Masks = new int[L][];
            for (int i = 0; i < L; ++i)
                Masks[i] = new int[Count];

            Lenghts = new int[Count];

            using (StreamReader file = new StreamReader(path))
            {
                string line;
                for (int mask = 0; mask < Count; ++mask)
                {
                    line = file.ReadLine();

                    int address = 0;
                    var parts = line.Split(new char[] { ';', '.' });
                    Lenghts[mask] = int.Parse(parts[6]);

                    for (int i = 0; i < 4; ++i)
                        address |= int.Parse(parts[2 + i]) << (8 * (3 - i));

                    //for (int l = 0; l < L; ++l)
                    //    Masks[l][mask] = (address >> (32 - rSums[l])) & ((2 << R[l] - 1) - 1);
                }
            }
        }

        public void PrintMasks()
        {
            StreamWriter outFile = new StreamWriter("outFile.txt");

            for (int mask = 0; mask < Count; ++mask)
            {
                for (int l = 0; l < L; ++l)
                    outFile.Write("{0} . ", Convert.ToString(Masks[l][mask], 2).PadLeft(R[l], '0'));

                outFile.WriteLine(Lenghts[mask]);
            }

            outFile.Close();
        }

        public void Build()
        {
            //Marking borders between nodes
            int[][] nodesBorders = new int[L][];
            int[][] nodesIndexes = new int[L][];
            for (int l = 0; l < L; ++l)
            {
                nodesBorders[l] = new int[Count];
                nodesBorders[l][0] = 1;

                nodesIndexes[l] = new int[Count];
            }

            for (int l = 1; l < L; ++l)
            {
                //To będzie rozproszone na bloki i wątki
                for (int i = 1; i < Count; ++i)
                    if (Masks[l - 1][i - 1] != Masks[l - 1][i] || nodesBorders[l - 1][i] == 1)
                        nodesBorders[l][i] = 1;
            }

            //Counting number of nodes and indexing them on each level. Indexing is done from 1 up, since 0 means empty value
            LevelsSizes = new int[L];

            for (int l = 0; l < L; ++l)
            {
                //To będzie zrobione Thrustem. Najpierw inclusive scan, potem pomnożenie przez nodesBorders, żeby wróciły 0.
                nodesIndexes[l] = nodesBorders[l].Scan((a, b) => a + b).Zip(nodesBorders[l], (a, b) => a * b).ToArray();

                //To można uzyskać z inclusive scan
                LevelsSizes[l] = nodesIndexes[l].Max();
            }

            //Filling start and end indexes of tree nodes
            var startIndexes = new int[L][];
            var endIndexes = new int[L][];

            for (int l = 0; l < L; ++l)
            {
                startIndexes[l] = new int[LevelsSizes[l]];
                startIndexes[l][0] = 0;

                endIndexes[l] = new int[LevelsSizes[l]];
                endIndexes[l][LevelsSizes[l] - 1] = Count;
            }

            for (int l = 1; l < L; ++l)
            {
                //To będzie rozporoszone na bloki i wątki
                for (int i = 1; i < Count; ++i)
                    if (nodesIndexes[l][i] > 0)
                    {
                        startIndexes[l][nodesIndexes[l][i] - 1] = i;
                        endIndexes[l][nodesIndexes[l][i] - 2] = i;
                    }
            }

            //Removing empty nodes
            for (int l = 0; l < L; ++l)
            {
                int[] toLeave = new int[LevelsSizes[l]];

                //To będzie rozproszone na bloki
                for (int node = 0; node < LevelsSizes[l]; ++node)
                {
                    //To będzie rozproszone na wątki. 1 będzie wpisywana do pamięci dzielonej i co iterację sprawdzana.
                    for (int i = startIndexes[l][node]; i < endIndexes[l][node]; ++i)
                    {
                        if (Lenghts[i] > rPreSums[l])
                        {
                            toLeave[node] = 1;
                            break;
                        }
                    }
                }

                //Zmniejszanie levelu. Najpierw inclusive scan, potem pomnożenie przez toLeave, żeby wróciły 0.
                int[] newIndexes = toLeave.Scan((a, b) => a + b).Zip(toLeave, (a, b) => a * b).ToArray();
                int[] newStartIndexes = new int[newIndexes.Max()];
                int[] newEndIndexes = new int[newIndexes.Max()];

                //To będzie rozproszone na bloki i wątki
                for (int node = 0; node < LevelsSizes[l]; ++node)
                {
                    if (newIndexes[node] != 0)
                    {
                        newStartIndexes[newIndexes[node] - 1] = startIndexes[l][node];
                        newEndIndexes[newIndexes[node] - 1] = endIndexes[l][node];
                    }
                    else
                    {
                        nodesBorders[l][startIndexes[l][node]] = 0;
                    }
                }

                nodesIndexes[l] = nodesBorders[l].Scan((a, b) => a + b).Zip(nodesBorders[l], (a, b) => a * b).ToArray();
                startIndexes[l] = newStartIndexes;
                endIndexes[l] = newEndIndexes;
                LevelsSizes[l] = newIndexes.Max();
            }


            //Filling children of tree nodes
            ChildrenCount = new int[L];
            for (int l = 0; l < L; ++l)
                ChildrenCount[l] = 2 << (R[l] - 1);

            Children = new int[L - 1][];
            for (int l = 0; l < L - 1; ++l)
                Children[l] = new int[LevelsSizes[l] * ChildrenCount[l]];

            for (int l = 0; l < L - 1; ++l)
                //Idziemy po węzłach na poziomie (to będzie rozproszone na bloki)
                for (int node = 0; node < LevelsSizes[l]; ++node)
                    //Szukamy dzieci danego węzła. To będzie rozproszone na wątki
                    for (int i = startIndexes[l][node]; i < endIndexes[l][node]; ++i)
                        if (nodesIndexes[l + 1][i] > 0)
                            Children[l][node * ChildrenCount[l] + Masks[l][i]] = nodesIndexes[l + 1][i];

            ////Building lists of items for each node
            ListItems = new int[Count];

            ListsStarts = new int[L][];
            ListsLenghts = new int[L][];

            for (int l = 0; l < L; ++l)
                ListsLenghts[l] = new int[LevelsSizes[l]];

            //Filling list lenghts
            for (int l = 0; l < L; ++l)
                //Idziemy po węzłach na poziomie (to będzie rozproszone na bloki i wątki)
                for (int node = 0; node < LevelsSizes[l]; ++node)
                {
                    int lenght = 0;
                    for (int i = startIndexes[l][node]; i < endIndexes[l][node]; ++i)
                        if (Lenghts[i] > rPreSums[l] && Lenghts[i] <= rSums[l])
                            ++lenght;

                    ListsLenghts[l][node] = lenght;
                }

            ////Filling lists start indexes
            int[] totalListItemsPerLevel = new int[L];
            for (int l = 0; l < L; ++l)
            {
                //To będzie robione Thrustem
                ListsStarts[l] = ListsLenghts[l].PreScan((a, b) => a + b, 0).ToArray();

                totalListItemsPerLevel[l] = ListsLenghts[l].Sum();
            }

            ////Shifting lists
            int shift = 0;
            for (int l = 1; l < L; ++l)
            {
                shift += totalListItemsPerLevel[l - 1];

                //To będzie robione Thrustem
                ListsStarts[l] = ListsStarts[l].Select(x => x + shift).Zip(ListsLenghts[l], (a, b) => b != 0 ? a : 0).ToArray();
            }

            ////Filling lists items
            for (int l = 0; l < L; ++l)
                //To będzie rozproszone na bloki i wątki (każde wątek wypełnia jeden węzeł)
                for (int node = 0; node < LevelsSizes[l]; ++node)
                {
                    int insertShift = 0;
                    for (int maskLenght = rSums[l]; maskLenght > rPreSums[l]; --maskLenght)
                    {
                        for (int i = startIndexes[l][node]; i < endIndexes[l][node]; ++i)
                            if (Lenghts[i] == maskLenght)
                            {
                                ListItems[ListsStarts[l][node] + insertShift] = i;
                                ++insertShift;
                            }
                    }
                }
        }

        public int[] Match(int[][] ips, int count)
        {
            int [] result = new int[count];
            
            //TODO: Masks to ips w wyszukiwaniu węzłów
            //To będzie rozproszone na bloki i wątki. NodesToCheck musi być osobne dla każdego wątku
            for (int i = 0; i < count; ++i)
            {
                //Find nodes to be searched
                int[] nodesToCheck = new int[L];
                nodesToCheck[0] = 1;
                for (int l = 1; l < L; ++l)
                    if (nodesToCheck[l - 1] != 0)
                        nodesToCheck[l] = Children[l - 1][(nodesToCheck[l - 1] - 1) * ChildrenCount[l-1] + ips[l - 1][i]];
                    else
                        break;

                //Search lists
                for (int l = L - 1; l >= 0 && result[i] == 0; --l)
                    if (nodesToCheck[l] != 0)
                    {
                        for (int s = ListsStarts[l][nodesToCheck[l] - 1];
                            s < ListsStarts[l][nodesToCheck[l] - 1] + ListsLenghts[l][nodesToCheck[l] - 1] && result[i] == 0;
                            ++s)
                        {
                            int shitf = R[l] - (Lenghts[ListItems[s]] - rPreSums[l]);
                            if (Masks[l][ListItems[s]] >> shitf == ips[l][i] >> shitf)
                                result[i] = 1;
                        }
                    }
            }
            return result;
        }
    }
}
