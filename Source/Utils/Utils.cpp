#include "Utils.h"

void SplitLine(const string& str, const string& delim, vector<string>& parts)
{
	size_t start, end = 0;
	while (end < str.size())
	{
		start = end;
		while (start < str.size() && (delim.find(str[start]) != string::npos))
			start++;

		end = start;
		while (end < str.size() && (delim.find(str[end]) == string::npos))
			end++;

		if (end - start != 0)
			parts.push_back(string(str, start, end - start));
	}
}