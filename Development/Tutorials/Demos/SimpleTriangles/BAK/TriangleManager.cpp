#include <limits>
#include <assimp.h>
#include <aiScene.h>
#include <aiPostProcess.h> 
#include "TriangleManager.h"

inline uint interleaveBits(uint3 input)
{
	input.x = (input.x | (input.x << 16)) & 0x030000FF;
	input.x = (input.x | (input.x <<  8)) & 0x0300F00F;
	input.x = (input.x | (input.x <<  4)) & 0x030C30C3;
	input.x = (input.x | (input.x <<  2)) & 0x09249249;

	input.y = (input.y | (input.y << 16)) & 0x030000FF;
	input.y = (input.y | (input.y <<  8)) & 0x0300F00F;
	input.y = (input.y | (input.y <<  4)) & 0x030C30C3;
	input.y = (input.y | (input.y <<  2)) & 0x09249249;

	input.z = (input.z | (input.z << 16)) & 0x030000FF;
	input.z = (input.z | (input.z <<  8)) & 0x0300F00F;
	input.z = (input.z | (input.z <<  4)) & 0x030C30C3;
	input.z = (input.z | (input.z <<  2)) & 0x09249249;

	return (input.x | (input.y << 1) | (input.z << 2));
}

void TriangleManager::loadMesh(const char *filename)
{
	const aiScene *scene = aiImportFile(filename, aiProcessPreset_TargetRealtime_Fast);

	if (!scene)
	{
		std::cerr << aiGetErrorString() << std::endl;
		return;
	}

	for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
	{
		const aiMesh *mesh = scene->mMeshes[meshIndex];

		for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
		{
			const aiVector3D *vertex = &mesh->mVertices[vertexIndex];
			mPositionList.push_back(make_float3(vertex->x, vertex->y, vertex->z));
		}

		unsigned int faceOffset = mPositionList.size();

		for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
		{
			const struct aiFace *face = &mesh->mFaces[faceIndex];

			uint3 f;
			f.x = faceOffset + face->mIndices[0];
			f.y = faceOffset + face->mIndices[1];
			f.z = faceOffset + face->mIndices[2];
			mFaceList.push_back(f);
		}
	}

	std::cout << "Position List: " << mPositionList.size() << std::endl;
	std::cout << "Face List: " << mFaceList.size() << std::endl;
}

void TriangleManager::generatePools()
{
	// compute the global bounding box
	BoundingBox bb;

	for (unsigned int i = 0; i < mPositionList.size(); ++i)
	{
		bb.min = min(bb.min, mPositionList[i]);
		bb.max = max(bb.max, mPositionList[i]);
	}

	float3 bbSize = bb.max - bb.min;
	float3 bbMaxSize = make_float3(max(bbSize.x, max(bbSize.y, bbSize.z)));

	// normalize the mesh
	for (unsigned int i = 0; i < mPositionList.size(); ++i)
	{
		mPositionList[i] = (mPositionList[i] - bb.min) / bbMaxSize;
	}

	std::map< uint, uint > trianglesLoc;

	for (unsigned int i = 0; i < mFaceList.size(); ++i)
	{
		const uint3 &face = mFaceList[i];

		// compute the triangle's bounding box
		BoundingBox bb;
		bb.min = min(mPositionList[face.x], min(mPositionList[face.y], mPositionList[face.z]));
		bb.max = max(mPositionList[face.x], max(mPositionList[face.y], mPositionList[face.z]));

		// hash the position of the bounding box's center
		uint3 bbCenterI = make_uint3(bb.getCenter() * 0x000003FF);
		
		uint posHash = interleaveBits(make_uint3(bbCenterI.x & 0x000003FF, bbCenterI.y & 0x000003FF, bbCenterI.z & 0x000003FF));
		trianglesLoc[posHash] = i;
	}

	unsigned int i = 0;

	while (i < trianglesLoc.size())
	{

	}
}