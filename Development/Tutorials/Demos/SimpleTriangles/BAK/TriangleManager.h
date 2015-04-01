#ifndef _TRIANGLEMANAGER_H_
#define _TRIANGLEMANAGER_H_

#include <vector>
#include <gigavoxels/core/vector_types_ext.h>

struct BoundingBox
{
	BoundingBox()
	{
		min = make_float3(+std::numeric_limits<float>::max());
		max = make_float3(-std::numeric_limits<float>::max());
	}

	float3 getCenter() const
	{
		return min + 0.5f * (max - min);
	}

	float3 min;
	float3 max;
};

class TriangleManager
{
	enum { PageSize = 32 };

public:
	void loadMesh(const char *filename);
	void generatePools();

private:
	std::vector<float3> mPositionList;
	std::vector<uint3> mFaceList;
};
#endif // !_TRIANGLEMANAGER_H_