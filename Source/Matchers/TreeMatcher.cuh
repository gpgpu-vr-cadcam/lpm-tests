#pragma once

#include "BaseMatcher.cuh"

class TreeMatcher : public BaseMatcher
{
public:
	void BuildModel(IPSet set) override;
	void Match(IPSet set) override;
};
